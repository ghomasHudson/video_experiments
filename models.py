'''Patch-based models'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from tools.model_tools import generate_spatial_mask, add_noise_and_clip
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from torch.nn.modules.transformer import _generate_square_subsequent_mask
from torch.nn import PixelShuffle, PixelUnshuffle


def load_model(args):
    """
    Model Loader for all the models contained within this file. Add new mappings here for external loading
    """
    model = None

    match args.model_name:
        case 'patch_decoder':
            model = PatchDecoder(**vars(args))
        case 'patch_decoder_a1':
            model = PatchDecoderA1(**vars(args))
        case 'patch_decoder_a2':
            model = PatchDecoderA2(**vars(args))
        case 'patch_decoder_a3':
            model = PatchDecoderA3(**vars(args))
        case 'patch_decoder_a4':
            model = PatchDecoderA4(**vars(args))
        case 'patch_decoder_u':
            model = PatchDecoderU(**vars(args))
        case _:
            model = PatchDecoderA3(**vars(args))

    return model


class PatchDecoderA1(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses auto-regressive patch decoding (patches predicted sequentially)
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_input_len, hidden_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos_dec = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(layers)])

        # Decoding layers
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.decoder_patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, 
                                                    in_channels=channels)
        self.decoder_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.patch_decoder = nn.ModuleList([
            TransformerBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, 
                             dropout=dropout, final_dropout=True, last_residual=True)
            for _ in range(decoding_layers)])

        # Output layer
        self.out_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                          head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                          last_residual=False)
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # If targets are provided, split targets into patches
        if targets is not None:
            targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
            targets = targets.view(b * t, n, c, p, p)
            # Remove last patch from targets as not needed to generate next frame
            targets = targets[:, :-1, ...]

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.type_as(x)[:, :t, :]
        
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)
        x = x + self.spatial_pos.type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, attn = layer(x, (b, t, n, k), cache_attn=cache_attn)

        # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # Generate start of sequence token
        start = self.start_token.expand(b * t, -1, -1).type_as(x)

        # If targets, run training mode with teacher forcing
        if targets is not None:
            # Add random noise to half of the target patches at random
            targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
            # Embed target patches using fixed patch embedder (do not update or learn)
            targets = self.decoder_patch_embedder(targets.reshape(-1, c, p, p))
            targets = targets.view(b * t, n - 1, k)

            # Add spatial positional encoding
            targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]

            # Concatenate start token and targets to form decoder input
            targets = torch.cat([start, targets], dim=1)
            
            x = self.decoder_combine(torch.cat([x, targets], dim=-1))
            
            # Pass through decoder blocks
            for decoder_layer in self.patch_decoder:
                x, _ = decoder_layer(x, x, cache_attn=cache_attn)
            
            # Project to output dimension
            x, _ = self.out_layer(x, x, cache_attn=cache_attn)
            out = self.out_activation(x)

        # If no targets (eval), run autoregressive inference mode
        else:
            # Initialize list to collect outputs
            outputs = []

            # Generate patches one by one
            for i in range(n):
                if i == 0 or cache_decoder_attn is False:
                    decoder_x = start.clone()
                if i == 0:
                    decoder_x = self.decoder_combine(torch.cat([x[:, 0, :].unsqueeze(1), decoder_x], dim=-1))

                # Pass through decoder layers
                for decoder_layer in self.patch_decoder:
                    decoder_x, _ = decoder_layer(decoder_x, decoder_x, cache_attn=cache_decoder_attn)
                decoder_x, _ = self.out_layer(decoder_x, decoder_x, cache_attn=cache_decoder_attn)

                # Get the last output token
                out_patch = decoder_x[:, -1, :].unsqueeze(1)
                out_patch = self.out_activation(out_patch)
                # Collect output
                outputs.append(out_patch.clone())
                # Prepare next decoder input if not last patch
                if i < n - 1:
                    # Embed target patches using fixed patch embedder (do not update or learn)
                    out_patch = self.decoder_patch_embedder(out_patch.view(b * t, c, p, p))
                    out_patch = out_patch.reshape(b * t, 1, k)
                    pos_encoding = self.spatial_pos_dec[:, i, :].type_as(out_patch)
                    out_patch = out_patch + pos_encoding  # Add positional encoding
                    out_patch = self.decoder_combine(torch.cat([x[:, i + 1, :].unsqueeze(1), out_patch], dim=-1))
                    # If caching (inference), use out_patch as start, else replace start with sequence
                    if cache_decoder_attn:
                        decoder_x = out_patch
                    else:
                        start = torch.cat([start, out_patch], dim=1)

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()
            
            # Combine all outputs
            out = torch.cat(outputs, dim=1)  # Shape: [b * t, n - 1, output_dim]

        # Reshape and combine patches to form images
        out = out.reshape(b * t, n, c, p, p)
        out = combine_tensor_patches(out, original_size=h, window_size=p, stride=p)
        out = out.view(b, t, c, h, w)

        return {'predictions': out}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()
        for layer in self.patch_decoder:
            layer.reset_cache()

        self.out_layer.reset_cache()


class PatchDecoderA2(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses auto-regressive patch decoding (patches predicted sequentially)
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)
        self.patch_embedder_dec = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions with random initialization
        self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(layers)])

        # Decoding layers
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.decoder_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.patch_decoder = nn.ModuleList([
            TransformerBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, 
                             dropout=dropout, final_dropout=True, last_residual=True)
            for _ in range(decoding_layers)])

        # Output layer
        self.out_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                          head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                          last_residual=False)
        self.out_activation = nn.Sigmoid()

    def forward(self, x, padding_mask=None, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # If targets are provided, split targets into patches
        if targets is not None:
            targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
            targets = targets.view(b * t, n, c, p, p)
            # Remove last patch from targets as not needed to generate next frame
            targets = targets[:, :-1, ...]

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)

        # Add spatial positional encoding
        x = x + self.spatial_pos.type_as(x)

        # add global token to patch dimension
        global_token = self.global_token.expand(b * t, -1, -1).type_as(x)
        x = torch.cat([global_token, x], dim=1)
        # redefine n
        n = n + 1
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.type_as(x)[:, :t, :]
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)

        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, global_only=False)

        # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # remove global tokens from patches and store as x_global
        x_global = x[:, 0, :].reshape(b, t, k)
        # repeat t times in batch dimension
        # x_global = x_global.repeat_interleave(t, dim=0)
        x = x[:, 1:, :]
        # redefine n
        n = n - 1

        # Generate start of sequence token
        start = self.start_token.expand(b * t, -1, -1).type_as(x)

        # If targets, run training mode with teacher forcing
        if targets is not None:
            # Add random noise to half of the target patches at random
            targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
            # Embed target patches
            targets = self.patch_embedder_dec(targets.reshape(-1, c, p, p))
            targets = targets.view(b * t, n - 1, k)
            # Add spatial positional encoding
            targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]
            targets = torch.cat([start, targets], dim=1)

            # concatenate x and targets
            x = torch.cat([x, targets], dim=-1)
            x = self.decoder_combine(x)
            
            for decoder_layer in self.patch_decoder:
                x, _ = decoder_layer(x, x) 
            # Project to output dimension
            x, _ = self.out_layer(x, x)
            x = self.out_activation(x)
            
        # If no targets (eval), run autoregressive inference mode
        else:
            # Initialize list to collect outputs
            outputs = []

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()
            
            # Generate patches one by one
            for i in range(n):
                if i == 0:
                    target = start.clone()
                    target = torch.cat([x[:, 0, :].unsqueeze(1), target], dim=2)
                    target = self.decoder_combine(target)
                    target_x = target

                # Pass through decoder layers
                for decoder_layer in self.patch_decoder:
                    target_x, _ = decoder_layer(target_x, target_x, cache_attn=cache_decoder_attn)
                target_x, _ = self.out_layer(target_x, target_x, cache_attn=cache_decoder_attn)

                # Get the last output token
                out_patch = target_x[:, -1, :].unsqueeze(1)
                out_patch = self.out_activation(out_patch)
                # Collect output
                outputs.append(out_patch.clone())
                # Prepare next decoder input if not last patch
                if i < n - 1:
                    # Embed target patches using fixed patch embedder (do not update or learn)
                    out_patch = self.patch_embedder_dec(out_patch.view(b * t, c, p, p))
                    out_patch = out_patch.reshape(b * t, 1, k)
                    pos_encoding = self.spatial_pos_dec[:, i, :].type_as(out_patch).unsqueeze(1)
                    out_patch = out_patch + pos_encoding  # Add positional encoding
                    # combine x and out_patch
                    out_patch = torch.cat([x[:, i + 1, :].unsqueeze(1), out_patch], dim=2)
                    out_patch = self.decoder_combine(out_patch)

                    if cache_decoder_attn:
                        target_x = out_patch
                    else:
                        target_x = torch.cat([target, out_patch], dim=1)
            
            # Combine all outputs
            x = torch.cat(outputs, dim=1)  # Shape: [b * t, n - 1, output_dim]

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()
        for layer in self.patch_decoder:
            layer.reset_cache()
        self.out_layer.reset_cache()


class PatchDecoderA3(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses auto-regressive patch decoding (patches predicted sequentially) with global token
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)
        self.patch_embedder_dec = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions with random initialization
        self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(layers)])
        
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.patch_decoder = nn.ModuleList([
            DecoderCrossAttentionBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, 
                             dropout=dropout, final_dropout=True, last_residual=True)
            for _ in range(decoding_layers)])

        # Output layer
        self.out_layer = DecoderCrossAttentionBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                          head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                          last_residual=False)
        self.out_activation = nn.Sigmoid()

    def forward(self, x, padding_mask=None, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # If targets are provided, split targets into patches
        if targets is not None:
            targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
            targets = targets.view(b * t, n, c, p, p)
            # Remove last patch from targets as not needed to generate next frame
            targets = targets[:, :-1, ...]

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)

        # Add spatial positional encoding
        x = x + self.spatial_pos.type_as(x)

        # add global token to patch dimension
        global_token = self.global_token.expand(b * t, -1, -1).type_as(x)
        x = torch.cat([global_token, x], dim=1)
        # redefine n
        n = n + 1
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.type_as(x)[:, :t, :]
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)

        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, global_only=False)

        # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # remove global tokens from patches and store as x_global
        x_global = x[:, 0, :].reshape(b, t, k)
        # repeat t times in batch dimension
        # x_global = x_global.repeat_interleave(t, dim=0)
        x = x[:, 1:, :]
        # redefine n
        n = n - 1

        # Generate start of sequence token
        start = self.start_token.expand(b * t, -1, -1).type_as(x)
        
        # If targets, run training mode with teacher forcing
        if targets is not None:
            # Add random noise to half of the target patches at random
            targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
            # Embed target patches
            targets = self.patch_embedder_dec(targets.reshape(-1, c, p, p))
            targets = targets.view(b * t, n - 1, k)
            # Add spatial positional encoding
            targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]
            targets = torch.cat([start, targets], dim=1)
            
            for decoder_layer in self.patch_decoder:
                x, _ = decoder_layer(x, x, targets) 
            # Project to output dimension
            x, _ = self.out_layer(x, x, targets)
            x = self.out_activation(x)
            
        # If no targets (eval), run autoregressive inference mode
        else:
            # Initialize list to collect outputs
            outputs = []

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()
            
            # Generate patches one by one
            for i in range(n):
                if i == 0:
                    decoder_x = x[:, 0, :].unsqueeze(1)
                    target_x = start.clone()

                # Pass through decoder layers
                for decoder_layer in self.patch_decoder:
                    decoder_x, _ = decoder_layer(decoder_x, decoder_x, target_x, cache_attn=cache_decoder_attn)
                decoder_x, _ = self.out_layer(decoder_x, decoder_x, target_x, cache_attn=cache_decoder_attn)

                # Get the last output token
                out_patch = decoder_x[:, -1, :].unsqueeze(1)
                out_patch = self.out_activation(out_patch)
                # Collect output
                outputs.append(out_patch.clone())
                # Prepare next decoder input if not last patch
                if i < n - 1:
                    # Embed target patches using fixed patch embedder (do not update or learn)
                    out_patch = self.patch_embedder_dec(out_patch.view(b * t, c, p, p))
                    out_patch = out_patch.reshape(b * t, 1, k)
                    pos_encoding = self.spatial_pos_dec[:, i, :].type_as(out_patch).unsqueeze(1)
                    out_patch = out_patch + pos_encoding  # Add positional encoding

                    if cache_decoder_attn:
                        target_x = out_patch
                        decoder_x = x[:, i + 1, :].unsqueeze(1)
                    else:
                        target_x = torch.cat([target_x, out_patch], dim=1)
                        decoder_x = x[:, :i + 2, :]
            
            # Combine all outputs
            x = torch.cat(outputs, dim=1)  # Shape: [b * t, n - 1, output_dim]

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()
        for layer in self.patch_decoder:
            layer.reset_cache()
        self.out_layer.reset_cache()


class PatchDecoderA4(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses auto-regressive patch decoding (patches predicted sequentially), with global token in decoding.
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)
        self.patch_embedder_dec = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions with random initialization
        self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.global_cache = None
        
        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(layers)])
        
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.patch_decoder = nn.ModuleList([DecoderCrossAttentionBlockGlobal(
            hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
            final_dropout=True, last_residual=True) for _ in range(decoding_layers)])

        # Output layer
        self.out_layer = DecoderCrossAttentionBlockGlobal(
            hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
            final_dropout=False, last_residual=False)
        # Output activation
        self.out_activation = nn.Sigmoid()

    def forward(self, x, padding_mask=None, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # If targets are provided, split targets into patches
        if targets is not None:
            targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
            targets = targets.view(b * t, n, c, p, p)
            # Remove last patch from targets as not needed to generate next frame
            targets = targets[:, :-1, ...]

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)

        # Add spatial positional encoding
        x = x + self.spatial_pos.type_as(x)

        # add global token to patch dimension
        global_token = self.global_token.expand(b * t, -1, -1).type_as(x)
        x = torch.cat([global_token, x], dim=1)
        # redefine n
        n = n + 1
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.type_as(x)[:, :t, :]
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)

        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, global_only=False)

        # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # remove global tokens from patches and store as x_global
        x_global = x[:, 0, :].reshape(b, t, k)
        # repeat t times in batch dimension
        # x_global = x_global.repeat_interleave(t, dim=0)
        x = x[:, 1:, :]
        # redefine n
        n = n - 1

        if cache_attn and seq_idx is None:
            self.global_cache = x_global
        elif cache_attn and seq_idx is not None:
            x_global = torch.cat([self.global_cache, x_global], dim=1)
            self.global_cache = x_global

        t_g = x_global.size(1)
        # we have x_global of shape [b, t_g, k]
        # we want a mask for x_global of shape [b * t, n, t_g]
         
        # repeat t times in batch dimension
        x_global = x_global.repeat_interleave(t, dim=0)

        # timestep_indices = torch.arange(b * t_g, device=x.device) % t_g  # Shape: [b * t]
        # timesteps = torch.arange(t_g, device=x.device)  # Shape: [t]
        # mask = timestep_indices.unsqueeze(1) >= timesteps.unsqueeze(0)  # Shape: [b * t, t]
        # attn_mask = torch.zeros_like(mask)
        # attn_mask = attn_mask.masked_fill(~mask, float('-inf')).type_as(x).to(x.device)  # Shape: [b * t, t]
        # attn_mask = attn_mask.unsqueeze(1).expand(-1, n, -1)  # Shape: [b * t, n, t]
        # attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))

        attn_mask = torch.triu(torch.full((t_g, t), float('-inf'), device=x.device), diagonal=1)  # Shape: [t_g, t_g]
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n, -1)  # Shape: [t_g, n, t_g]
        attn_mask = attn_mask.unsqueeze(0).expand(b, -1, -1, -1)  # Shape: [b, t_g, n, t_g]
        attn_mask = attn_mask.reshape(b * t, n, t_g)

        # Generate start of sequence token
        start = self.start_token.expand(b * t, -1, -1).type_as(x)
        
        # If targets, run training mode with teacher forcing
        if targets is not None:
            # Add random noise to half of the target patches at random
            targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
            # Embed target patches
            targets = self.patch_embedder_dec(targets.reshape(-1, c, p, p))
            targets = targets.view(b * t, n - 1, k)
            # Add spatial positional encoding
            targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]
            targets = torch.cat([start, targets], dim=1)

            for decoder_layer in self.patch_decoder:
                x, _ = decoder_layer(x, x_global, targets, cross_attn_mask=attn_mask) 
            # Project to output dimension
            x, _ = self.out_layer(x, x_global, targets, cross_attn_mask=attn_mask)
            x = self.out_activation(x)
            
        # If no targets (eval), run autoregressive inference mode
        else:
            # Initialize list to collect outputs
            outputs = []

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()
            
            # Generate patches one by one
            for i in range(n):
                if i == 0:
                    decoder_x = x[:, 0, :].unsqueeze(1)
                    target_x = start.clone()
                    attn_mask_i = attn_mask[:, 0, :].unsqueeze(1)

                # Pass through decoder layers
                for decoder_layer in self.patch_decoder:
                    decoder_x, _ = decoder_layer(decoder_x, x_global, target_x, cache_attn=cache_decoder_attn,
                                                 cross_attn_mask=attn_mask_i)
                decoder_x, _ = self.out_layer(decoder_x, x_global, target_x, cache_attn=cache_decoder_attn,
                                              cross_attn_mask=attn_mask_i)

                # Get the last output token
                out_patch = decoder_x[:, -1, :].unsqueeze(1)
                out_patch = self.out_activation(out_patch)
                # Collect output
                outputs.append(out_patch.clone())
                # Prepare next decoder input if not last patch
                if i < n - 1:
                    # Embed target patches using fixed patch embedder (do not update or learn)
                    out_patch = self.patch_embedder_dec(out_patch.view(b * t, c, p, p))
                    out_patch = out_patch.reshape(b * t, 1, k)
                    pos_encoding = self.spatial_pos_dec[:, i, :].type_as(out_patch).unsqueeze(1)
                    out_patch = out_patch + pos_encoding  # Add positional encoding

                    if cache_decoder_attn:
                        target_x = out_patch
                        decoder_x = x[:, i + 1, :].unsqueeze(1)
                        attn_mask_i = attn_mask[:, i + 1, :].unsqueeze(1)
                    else:
                        target_x = torch.cat([target_x, out_patch], dim=1)
                        decoder_x = x[:, :i + 2, :]
                        attn_mask_i = attn_mask[:, :i + 2, :]
            
            # Combine all outputs
            x = torch.cat(outputs, dim=1)  # Shape: [b * t, n - 1, output_dim]

            for layer in self.patch_decoder:
                layer.reset_cache()
            self.out_layer.reset_cache()

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()
        for layer in self.patch_decoder:
            layer.reset_cache()
        self.out_layer.reset_cache()
        self.global_cache = None


class PatchDecoder(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses non-autoregressive decoding (patches predicted in parallel)
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_input_len, hidden_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(layers)])

        # Decoding layers
        self.decoder_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                              head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                              last_residual=False)
        
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.expand(b * n, -1, -1).type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.expand(b * n, -1, -1).type_as(x)[:, :t, :]
        
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)
        x = x + self.spatial_pos.expand(b * t, -1, -1).type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn)

        # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # Run decoder layer
        x, _ = self.decoder_layer(x, x, cache_attn=cache_attn)
        x = self.out_activation(x)

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()

        self.decoder_layer.reset_cache()
        

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, h_merge=2, w_merge=2):
        super().__init__()
        self.h_merge = h_merge
        self.w_merge = w_merge
        self.proj = nn.Linear(in_features * h_merge * w_merge, out_features, bias=False)

    def forward(self, x):
        # x shape: [batch_size, h_patches, w_patches, embedding_dim]
        h_patches, w_patches = x.shape[1], x.shape[2]
        assert h_patches % self.h_merge == 0 and w_patches % self.w_merge == 0, "Cannot merge, h_patches or w_patches not divisible by merge factor"
        x = rearrange(
            x,
            'b (h h_merge) (w w_merge) k -> b h w (h_merge w_merge k)',
            h_merge=self.h_merge,
            w_merge=self.w_merge
        )
        x = self.proj(x)
        return x


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, h_split=2, w_split=2):
        super().__init__()
        self.h_split = h_split
        self.w_split = w_split
        self.proj = nn.Linear(in_features, out_features * h_split * w_split, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)  # Optional blending factor

    def forward(self, x, skip):
        # x shape: [batch_size, h_patches, w_patches, embedding_dim]
        x = self.proj(x)
        k = x.size(-1) // (self.h_split * self.w_split)
        x = rearrange(
            x,
            'b h w (h_split w_split k) -> b (h h_split) (w w_split) k',
            h_split=self.h_split,
            w_split=self.w_split,
            k=k
        )
        # Combine with skip connection using linear interpolation
        x = torch.lerp(skip.float(), x.float(), self.fac)
        return x

    
class PatchDecoderU(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses non-autoregressive decoding (patches predicted in parallel) with U-net style architecture
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size ** 2) // patch_dim
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_input_len, hidden_dim))

        # Note: We will generate spatial positions dynamically after token merging
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos1 = nn.Parameter(torch.zeros(1, num_patches // 4, hidden_dim))

        # Spatiotemporal attention layers
        self.spacetime_layers_1 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Token merge layer
        self.token_merge = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_split = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)

        # Spatiotemporal attention layers after merging
        self.spacetime_layers_2 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers
        self.spacetime_layers_3 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Decoding layers
        self.decoder_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                              head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                              last_residual=False)
        
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, seq_idx=None):
        b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Split into patches
        x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

        # Patch embedding
        x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
        # Store hidden dim as k
        k = x.size(-1)
        
        # Apply positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
        if seq_idx is not None:
            x = x + self.temporal_pos.expand(b * n, -1, -1).type_as(x)[:, seq_idx, :].unsqueeze(1)
        else:
            x = x + self.temporal_pos.expand(b * n, -1, -1).type_as(x)[:, :t, :]
        
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)
        x = x + self.spatial_pos.expand(b * t, -1, -1).type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers_1:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn)

        # Perform 2x2 token merge here. Input shape is [b * t, n, k]
        n = x.size(1)
        h_patches = w_patches = int(math.sqrt(n))
        assert h_patches * w_patches == n, "Number of patches is not a perfect square"

        # Reshape x to [b * t, h_patches, w_patches, k]
        x = x.view(b * t, h_patches, w_patches, k)

        # Apply token merging
        skip_x = x
        x = self.token_merge(x)  # x now has shape [b * t, h_patches // 2, w_patches // 2, k]

        # Update number of patches and reshape back to [b * t, n', k]
        h_patches, w_patches = x.shape[1], x.shape[2]
        n = h_patches * w_patches
        x = x.view(b * t, n, k)
        # Generate new spatial positional embeddings
        x = x + self.spatial_pos1.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_2:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn)
    #     # Ensure shape is temporally batched for patch decoding
        x = x.view(b * t, n, k)

        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_patches, w_patches, k)

        # Reshape skip_x to [b * t, h_patches * h_split, w_patches * w_split, k]
        h_split = self.token_split.h_split
        w_split = self.token_split.w_split
        skip_h_patches = h_patches * h_split
        skip_w_patches = w_patches * w_split
        skip_x = skip_x.view(b * t, skip_h_patches, skip_w_patches, k)

        # Perform token splitting back to original number of patches
        x = self.token_split(x, skip_x)  # x now has shape [b * t, skip_h_patches, skip_w_patches, k]

        # Reshape x back to [b * t, n_original, k]
        n_original = skip_h_patches * skip_w_patches
        x = x.view(b * t, n_original, k)
        n = n_original

        # Spatiotemporal layers after splitting
        for layer in self.spacetime_layers_3:
            x, _ = layer(x, (b, t, n_original, k), cache_attn=cache_attn)

        # Run decoder layer
        x, _ = self.decoder_layer(x, x, cache_attn=cache_attn)
        x = self.out_activation(x)

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers_1:
            layer.reset_cache()

        self.decoder_layer.reset_cache()


class DecoderCrossAttentionBlockGlobal(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout

        # Pre-attention layer norm
        self.pre_norm = nn.LayerNorm(self.in_dim)

        # Self attention layer
        self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                             dropout=dropout)
        # Post-attention dropout and layernorm
        self.dropout_attn = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(self.in_dim)

        # Cross attention layer
        self.cross_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                   dropout=dropout)
        # Post-cross-attention dropout and layernorm
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.post_cross_norm = nn.LayerNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

        # self.out_block = TransformerBlock(self.in_dim, self.out_dim, self.out_dim, heads=heads, head_dim=head_dim,
        #                                   dropout=dropout, final_dropout=final_dropout, last_residual=last_residual)

    def forward(self, x, memory, targets, padding_mask=None, cache_attn=False, cross_attn_mask=None):
        # Pre-attention layer norm
        x = self.pre_norm(x)

        # Apply causal self attention
        x = x + self.dropout_attn(self.attn(x, memory, memory, need_weights=False, is_causal=True,
                                            padding_mask=padding_mask, cache_attn=False, attn_mask=cross_attn_mask)[0])
        x = self.post_norm(x)

        # Apply cross attention
        x = x + self.dropout_cross_attn(self.cross_attn(x, targets, targets, need_weights=False, is_causal=True,
                                                        cache_attn=cache_attn)[0])
        x = self.post_cross_norm(x)

        # Feedforward layers
        if self.last_residual and self.in_dim == self.out_dim:
            x = x + self.fc2(self.dropout_fc(self.activation(self.fc1(x))))
        else:
            x = self.fc2(self.dropout_fc(self.activation(self.fc1(x))))

        # x, _ = self.out_block(x, x, cache_attn=cache_attn)
        
        return x, None

    def reset_cache(self):
        self.attn.reset_cache()
        self.cross_attn.reset_cache()

        

class DecoderCrossAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout

        # Pre-attention layer norm
        self.pre_norm = nn.LayerNorm(self.in_dim)

        # Self attention layer
        self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                             dropout=dropout)
        # Post-attention dropout and layernorm
        self.dropout_attn = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(self.in_dim)

        # Cross attention layer
        self.cross_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                   dropout=dropout)
        # Post-cross-attention dropout and layernorm
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.post_cross_norm = nn.LayerNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

        # self.out_block = TransformerBlock(self.in_dim, self.out_dim, self.out_dim, heads=heads, head_dim=head_dim,
        #                                   dropout=dropout, final_dropout=final_dropout, last_residual=last_residual)

    def forward(self, x, memory, targets, padding_mask=None, cache_attn=False, cross_attn_mask=None):
        # Pre-attention layer norm
        x = self.pre_norm(x)

        # Apply causal self attention
        x = x + self.dropout_attn(self.attn(x, memory, memory, need_weights=False, is_causal=True,
                                            padding_mask=padding_mask, cache_attn=cache_attn)[0])
        x = self.post_norm(x)

        # Apply cross attention
        x = x + self.dropout_cross_attn(self.cross_attn(x, targets, targets, need_weights=False, is_causal=True,
                                                        cache_attn=cache_attn)[0])
        x = self.post_cross_norm(x)

        # Feedforward layers
        if self.last_residual and self.in_dim == self.out_dim:
            x = x + self.fc2(self.dropout_fc(self.activation(self.fc1(x))))
        else:
            x = self.fc2(self.dropout_fc(self.activation(self.fc1(x))))

        # x, _ = self.out_block(x, x, cache_attn=cache_attn)
        
        return x, None

    def reset_cache(self):
        self.attn.reset_cache()
        self.cross_attn.reset_cache()
        

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout

        # Pre-attention layer norm
        self.pre_norm = nn.LayerNorm(self.in_dim)

        # Self attention layer
        self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                             dropout=dropout)
        # Post-attention dropout and layernorm
        self.dropout_attn = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, memory, cache_attn=False):
        # Pre-attention layer norm
        x = self.pre_norm(x)

        # Apply causal self attention
        x = x + self.dropout_attn(self.attn(x, x, x, need_weights=False, is_causal=True,
                                            cache_attn=cache_attn)[0])
        x = self.post_norm(x)

        # Feedforward layers
        if self.last_residual and self.in_dim == self.out_dim:
            x = x + self.fc2(self.dropout_fc(self.activation(self.fc1(x))))
        else:
            x = self.fc2(self.dropout_fc(self.activation(self.fc1(x))))

        return x, None

    def reset_cache(self):
        self.attn.reset_cache()
    

class SpaceTimeBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True, torch_attn=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout
        self.torch_attn = torch_attn

        # Set head dimension default if not provided
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Pre-attention layer norm
        self.pre_norm = nn.LayerNorm(self.in_dim)

        # Spatial and temporal attention layers given attenton type
        self.spatial_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                                     dropout=dropout)
        self.temporal_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                      dropout=dropout)
       
        # Post-attention dropout and layernorm
        self.dropout_space_attn = nn.Dropout(dropout)
        self.dropout_time_attn = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, size, padding_mask=None, return_attn=False, cache_attn=False, global_only=False):            
        spatial_attn = None
        temporal_attn = None
        b, t, n, k = size
    
        # Pre-attention layer norm
        x = self.pre_norm(x)
            
        x = x + self.dropout_space_attn(self.spatial_attn(x, x, x, need_weights=False, is_causal=False, 
                                                          cache_attn=False)[0])
        if global_only:
            x_g = x[:, 0, :].unsqueeze(1)
            x_g = rearrange(x_g, ('(b t) n k -> (b n) t k'), b=b, n=1, t=t, k=k)
            x_g = x_g + self.dropout_time_attn(self.temporal_attn(x_g, x_g, x_g, need_weights=False, is_causal=True,
                                                                  cache_attn=cache_attn, padding_mask=padding_mask)[0])
            x_g = rearrange(x_g, ('(b n) t k -> (b t) n k'), b=b, n=1, t=t, k=k)
            x = torch.cat([x_g, x[:, 1:, :]], dim=1)
            
        else:
            x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
            x = x + self.dropout_time_attn(self.temporal_attn(x, x, x, need_weights=False, is_causal=True, 
                                                              cache_attn=cache_attn, padding_mask=padding_mask)[0])
            x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)

        # Post-attention layer norm
        x = self.post_norm(x)

        # Feedforward layers
        if self.last_residual and self.in_dim == self.out_dim:
            x = x + self.fc2(self.dropout_fc(self.activation(self.fc1(x))))
        else:
            x = self.fc2(self.dropout_fc(self.activation(self.fc1(x))))

        return x, {'attn_s': spatial_attn, 'attn_t': temporal_attn}

    def reset_cache(self):
        self.temporal_attn.reset_cache()


class CachedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, batch_first=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.key_cache = None
        self.value_cache = None

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, 
                                          batch_first=batch_first)

    def forward(self, query, key, value, need_weights=False, is_causal=False, cache_attn=False, attn_mask=None, 
                padding_mask=None):
        if cache_attn:
            if self.key_cache is None:
                self.key_cache = key.clone()
                self.value_cache = value.clone()
            else:
                self.key_cache = torch.cat((self.key_cache, key.clone()), dim=1)
                self.value_cache = torch.cat((self.value_cache, value.clone()), dim=1)

            key = self.key_cache.clone()
            value = self.value_cache.clone()

        if attn_mask is None:
            # Generate attention mask if causal and query size is greater than 1
            if is_causal and query.size(1) > 1:
                tgt_len = query.size(1)
                src_len = key.size(1)
                attn_mask = torch.triu(torch.full((tgt_len, src_len), float('-inf'), device=query.device), diagonal=1)
            else:
                attn_mask = None
        else:
            # attn mask is [batch_size, n, t], expand to [batch_size, num_heads, n, t]
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        # if padding_mask is not None:
        #     # assert mask is boolean
        #     assert padding_mask.dtype == torch.bool
            
        out = self.attn(query, key, value, need_weights=need_weights, attn_mask=attn_mask, 
                        key_padding_mask=padding_mask)[0]
        
        return out, None

    def reset_cache(self):
        self.key_cache = None
        self.value_cache = None


class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, projection_dim, in_channels):
        super().__init__()
        self.projection_dim = projection_dim
        self.encoder = nn.Conv2d(in_channels, projection_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch, channels, patch_size, patch_size = x.shape
        encoded = self.encoder(x)
        return encoded.reshape(batch, -1)


class FusionConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, kernel_size=3, padding=1):
        super(FusionConvLayer, self).__init__()
        layers = []
        current_in_channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x1, x2):
        """
        x1 and x2 are tensors of shape [batch_size, timesteps, channels, height, width]
        """
        # Concatenate along the channel dimension
        x = torch.cat([x1, x2], dim=2)  # Shape: [batch_size, timesteps, 2 * channels, height, width]
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)  # Merge batch and timestep dimensions
        x = self.conv(x)
        x = x.view(b, t, -1, h, w)  # Reshape back to [batch_size, timesteps, channels, height, width]
        return x


# class PatchDecoderArc3(nn.Module):
#     def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
#                  channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
#         super().__init__()

#         # Calculate patch and model dimensions
#         patch_dim = patch_size ** 2
#         num_patches = (img_size ** 2) // patch_dim
#         hidden_dim = patch_dim * embed_factor
#         head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

#         # Set model attributes
#         self.patch_size = patch_size
#         self.mc_drop = mc_drop
#         self.num_patches = num_patches

#         # Linear patch encoder
#         self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

#         # Learnable encodings for temporal and spatial positions with random initialization
#         self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
#         self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
#         self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
#         # global token embedding
#         self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
#         # Spatiotemporal attention layers
#         self.spacetime_layers = nn.ModuleList([
#             SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
#                            torch_attn=torch_attn) for _ in range(layers)])

#         # Decoding layers
#         self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
#         self.decoder_patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, 
#                                                     in_channels=channels)
#         self.decoder_combine = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.patch_decoder = nn.ModuleList([
#             DecoderCrossAttentionBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, 
#                                        dropout=dropout, final_dropout=True, last_residual=True)
#             for _ in range(decoding_layers)])

#         # Output layer
#         self.out_layer = DecoderCrossAttentionBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
#                                                     head_dim=head_dim, dropout=dropout, final_dropout=False, 
#                                                     last_residual=False)
#         self.out_activation = nn.Sigmoid()

#     def forward(self, x, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
#         b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
#         n, p = self.num_patches, self.patch_size  # Number of patches and patch size

#         # Split into patches
#         x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

#         # If targets are provided, split targets into patches
#         if targets is not None:
#             targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
#             targets = targets.view(b * t, n, c, p, p)
#             # Remove last patch from targets as not needed to generate next frame
#             targets = targets[:, :-1, ...]

#         # Patch embedding
#         x = self.patch_embedder(x.view(b * t * n, c, p, p)).view(b * t, n, -1)
#         # Store hidden dim as k
#         k = x.size(-1)

#         # Add spatial positional encoding
#         x = x + self.spatial_pos.type_as(x)

#         # add global token to patch dimension
#         global_token = self.global_token.expand(b * t, -1, -1).type_as(x)
#         x = torch.cat([global_token, x], dim=1)
#         # redefine n
#         n = n + 1
        
#         # Apply positional encodings
#         x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=t, k=k)
#         if seq_idx is not None:
#             x = x + self.temporal_pos.type_as(x)[:, seq_idx, :].unsqueeze(1)
#         else:
#             x = x + self.temporal_pos.type_as(x)[:, :t, :]
#         # rearrange back
#         x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)
        
#         # Spatiotemporal layers
#         for layer in self.spacetime_layers:
#             x, attn = layer(x, (b, t, n, k), cache_attn=cache_attn)

#         # Ensure shape is temporally batched for patch decoding
#         x = x.view(b * t, n, k)

#         # remove global tokens from patches and store as x_global
#         x_global = x[:, 0, :].reshape(b, t, k)
#         # repeat t times in batch dimension
#         x_global = x_global.repeat_interleave(t, dim=0)
        
#         x = x[:, 1:, :]
#         # redefine n
#         n = n - 1

#         # TODO change to broadcast to save memory
#         # Compute timestep indices for each sample
#         timestep_indices = torch.arange(b * t, device=x.device) % t  # Shape: [b * t]
#         timesteps = torch.arange(t, device=x.device)  # Shape: [t]
#         # Generate the mask: positions where timestep_indices >= timesteps
#         mask = timestep_indices.unsqueeze(1) >= timesteps.unsqueeze(0)  # Shape: [b * t, t]
#         attn_mask = torch.zeros_like(mask)
#         attn_mask = attn_mask.masked_fill(~mask, float('-inf')).type_as(x).to(x.device)  # Shape: [b * t, t]
#         # Expand attn_mask to match the query length (n)
#         attn_mask = attn_mask.unsqueeze(1).expand(-1, n, -1)  # Shape: [b * t, n, t]

#         # Generate start of sequence token
#         start = self.start_token.expand(b * t, -1, -1).type_as(x)

#         # If targets, run training mode with teacher forcing
#         if targets is not None:
#             # Add random noise to half of the target patches at random
#             targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
#             # Embed target patches
#             targets = self.decoder_patch_embedder(targets.reshape(-1, c, p, p))
#             targets = targets.view(b * t, n - 1, k)

#             # Add spatial positional encoding
#             targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]

#             # Concatenate start token and targets to form decoder input
#             targets = torch.cat([start, targets], dim=1)
            
#             x = self.decoder_combine(torch.cat([x, targets], dim=-1))
            
#             # Pass through decoder blocks
#             for decoder_layer in self.patch_decoder:
#                 x, _ = decoder_layer(x, x_global, cache_attn=cache_attn, cross_attn_mask=attn_mask)
            
#             # Project to output dimension
#             x, _ = self.out_layer(x, x_global, cache_attn=cache_attn, cross_attn_mask=attn_mask)
#             out = self.out_activation(x)

#         # If no targets (eval), run autoregressive inference mode
#         else:
#             # Initialize list to collect outputs
#             outputs = []
            
#             for layer in self.patch_decoder:
#                 layer.reset_cache()
#             self.out_layer.reset_cache()
            
#             # Generate patches one by one
#             for i in range(n):
#                 if i == 0 or cache_decoder_attn is False:
#                     decoder_x = start.clone()
#                 if i == 0:
#                     decoder_x = self.decoder_combine(torch.cat([x[:, 0, :].unsqueeze(1), decoder_x], dim=-1))

#                 if i == 0:
#                     attn_mask_i = attn_mask[:, i, :].unsqueeze(1)
#                 else:
#                     attn_mask_i = attn_mask[:, :i + 1, :]
    
#                 # Pass through decoder layers
#                 for decoder_layer in self.patch_decoder:
#                     decoder_x, _ = decoder_layer(decoder_x, x_global, cache_attn=cache_decoder_attn, 
#                                                  cross_attn_mask=attn_mask_i)
#                 decoder_x, _ = self.out_layer(decoder_x, x_global, cache_attn=cache_decoder_attn,
#                                               cross_attn_mask=attn_mask_i)

#                 # Get the last output token
#                 out_patch = decoder_x[:, -1, :].unsqueeze(1)
#                 out_patch = self.out_activation(out_patch)
#                 # Collect output
#                 outputs.append(out_patch.clone())
#                 # Prepare next decoder input if not last patch
#                 if i < n - 1:
#                     # Embed target patches using fixed patch embedder (do not update or learn)
#                     out_patch = self.decoder_patch_embedder(out_patch.view(b * t, c, p, p))
#                     out_patch = out_patch.reshape(b * t, 1, k)
#                     pos_encoding = self.spatial_pos_dec[:, i, :].type_as(out_patch).unsqueeze(1)
#                     out_patch = out_patch + pos_encoding  # Add positional encoding
#                     out_patch = self.decoder_combine(torch.cat([x[:, i + 1, :].unsqueeze(1), out_patch], dim=-1))
#                     # If caching (inference), use out_patch as start, else replace start with sequence
#                     if cache_decoder_attn:
#                         decoder_x = out_patch
#                     else:
#                         start = torch.cat([start, out_patch], dim=1)

#             for layer in self.patch_decoder:
#                 layer.reset_cache()
#             self.out_layer.reset_cache()
            
#             # Combine all outputs
#             out = torch.cat(outputs, dim=1)  # Shape: [b * t, n - 1, output_dim]

#         # Reshape and combine patches to form images
#         out = out.reshape(b * t, n, c, p, p)
#         out = combine_tensor_patches(out, original_size=h, window_size=p, stride=p)
#         out = out.view(b, t, c, h, w)

#         return {'predictions': out}

#     def reset_cache(self):
#         for layer in self.spacetime_layers:
#             layer.reset_cache()
#         for layer in self.patch_decoder:
#             layer.reset_cache()

#         self.out_layer.reset_cache()



# class DecoderCrossAttentionBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
#                  last_residual=True):
#         super().__init__()

#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_dim = hidden_dim
#         self.last_residual = last_residual
#         self.final_dropout = final_dropout

#         # Pre-attention layer norm
#         self.pre_norm = nn.LayerNorm(self.in_dim)

#         # Self attention layer
#         self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
#                                              dropout=dropout)
#         # Post-attention dropout and layernorm
#         self.dropout_attn = nn.Dropout(dropout)
#         self.post_norm = nn.LayerNorm(self.in_dim)

#         # Cross attention layer
#         self.cross_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
#                                                    dropout=dropout)
#         # Post-cross-attention dropout and layernorm
#         self.dropout_cross_attn = nn.Dropout(dropout)
#         self.post_cross_norm = nn.LayerNorm(self.in_dim)
        
#         # Feedforward layer
#         self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
#         self.activation = nn.GELU()
#         self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
#         self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

#     def forward(self, x, memory, cache_attn=False, cross_attn_mask=None):
#         # Pre-attention layer norm
#         x = self.pre_norm(x)

#         # Apply causal self attention
#         x = x + self.dropout_attn(self.attn(x, x, x, need_weights=False, is_causal=True,
#                                             cache_attn=cache_attn)[0])
#         x = self.post_norm(x)

#         # Apply cross attention
#         x = x + self.dropout_cross_attn(self.cross_attn(x, memory, memory, need_weights=False, is_causal=False,
#                                                         cache_attn=False, attn_mask=cross_attn_mask)[0])
#         x = self.post_cross_norm(x)

#         # Feedforward layers
#         if self.last_residual and self.in_dim == self.out_dim:
#             x = x + self.fc2(self.dropout_fc(self.activation(self.fc1(x))))
#         else:
#             x = self.fc2(self.dropout_fc(self.activation(self.fc1(x))))

#         return x, None

#     def reset_cache(self):
#         self.attn.reset_cache()

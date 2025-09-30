'''U-net style models for video prediction'''
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


def load_model_u(args):
    """
    Model Loader for all the models contained within this file. Add new mappings here for external loading
    """
    model = None

    if args.model_name == 'patch_decoder_u':
        model = PatchDecoderU(**vars(args))
    elif args.model_name == 'patch_decoder_u-large':
        model = PatchDecoderULarge(**vars(args))
    return model


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
        self.spatial_pos_0 = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos_1 = nn.Parameter(torch.zeros(1, num_patches // 4, hidden_dim))
        self.spatial_pos_2 = nn.Parameter(torch.zeros(1, num_patches // 16, hidden_dim))

        # Token merge/split layers
        self.token_merge_1 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_merge_2 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_split_2 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)
        self.token_split_1 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)

        # Spatiotemporal attention layers
        self.spacetime_layers_0 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers after merging
        self.spacetime_layers_1 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_2 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_3 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_4 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Final out layer
        self.out_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                          head_dim=head_dim, dropout=dropout, final_dropout=False, last_residual=False)
        
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, seq_idx=None, cache_decoder_attn=False):
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
        x = x + self.spatial_pos_0.expand(b * t, -1, -1).type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers_0:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn)

        # Perform 2x2 token merge here. Input shape is [b * t, n, k]
        n = x.size(1)
        h_0 = w_0 = int(math.sqrt(n))

        # Reshape x to [b * t, h_patches, w_patches, k]
        x = x.view(b * t, h_0, w_0, k)
        # Apply token merging
        skip_x1 = x
        x = self.token_merge_1(x)  # x now has shape [b * t, h_patches // 2, w_patches // 2, k]
        # Update number of patches and reshape back to [b * t, n', k]
        h_1, w_1 = x.shape[1], x.shape[2]
        n_1 = h_1 * w_1
        x = x.view(b * t, n_1, k)
        x = x + self.spatial_pos_1.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_1:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn)

        x = x.view(b * t, h_1, w_1, k)
        skip_x2 = x
        x = self.token_merge_2(x)
        h_2, w_2 = x.shape[1], x.shape[2]
        n_2 = h_2 * w_2
        x = x.view(b * t, n_2, k)
        x = x + self.spatial_pos_2.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_3:
            x, _ = layer(x, (b, t, n_2, k), cache_attn=cache_attn)
        
        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_2, w_2, k)
        skip_x2 = skip_x2.view(b * t, h_1, h_1, k)
        # Perform token splitting back to original number of patches
        x = self.token_split_2(x, skip_x2)  # x now has shape [b * t, skip_h_patches, skip_w_patches, k]
        # Reshape x back to [b * t, n_original, k]
        x = x.view(b * t, n_1, k)
        x = x + self.spatial_pos_1.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after splitting
        for layer in self.spacetime_layers_4:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn)

        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_1, w_1, k)
        skip_x1 = skip_x1.view(b * t, h_0, w_0, k)
        x = self.token_split_1(x, skip_x1)
        x = x.view(b * t, n, k)
        x = x + self.spatial_pos_0.expand(b * t, -1, -1).type_as(x)
        
        # Run decoder layer
        x, _ = self.out_layer(x, x, cache_attn=cache_attn)
        x = self.out_activation(x)

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers_0:
            layer.reset_cache()
        for layer in self.spacetime_layers_1:
            layer.reset_cache()
        for layer in self.spacetime_layers_2:
            layer.reset_cache()
        for layer in self.spacetime_layers_3:
            layer.reset_cache()
        for layer in self.spacetime_layers_4:
            layer.reset_cache()
        self.out_layer.reset_cache()


class PatchDecoderULarge(nn.Module):
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
        self.spatial_pos_0 = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos_1 = nn.Parameter(torch.zeros(1, num_patches // 4, hidden_dim))
        self.spatial_pos_2 = nn.Parameter(torch.zeros(1, num_patches // 16, hidden_dim))
        self.spatial_pos_0_dec = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos_1_dec = nn.Parameter(torch.zeros(1, num_patches // 4, hidden_dim))

        # Token merge/split layers
        self.token_merge_1 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_merge_2 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_split_2 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)
        self.token_split_1 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)

        # Spatiotemporal attention layers
        self.spacetime_layers_0 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers after merging
        self.spacetime_layers_1 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers
        self.spacetime_layers_2 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(8)])

        # Spatiotemporal attention layers
        self.spacetime_layers_3 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers
        self.spacetime_layers_4 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Final out layer
        self.out_layer = TransformerBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                          head_dim=head_dim, dropout=dropout, final_dropout=False, last_residual=False)
        
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, seq_idx=None, cache_decoder_attn=False):
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
        x = x + self.spatial_pos_0.expand(b * t, -1, -1).type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers_0:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, local_only=True)

        # Perform 2x2 token merge here. Input shape is [b * t, n, k]
        n = x.size(1)
        h_0 = w_0 = int(math.sqrt(n))

        # Reshape x to [b * t, h_patches, w_patches, k]
        x = x.view(b * t, h_0, w_0, k)
        # Apply token merging
        skip_x1 = x
        x = self.token_merge_1(x)  # x now has shape [b * t, h_patches // 2, w_patches // 2, k]
        # Update number of patches and reshape back to [b * t, n', k]
        h_1, w_1 = x.shape[1], x.shape[2]
        n_1 = h_1 * w_1
        x = x.view(b * t, n_1, k)
        x = x + self.spatial_pos_1.expand(b * t, -1, -1).type_as(x)
        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_1:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn, local_only=True)

        x = x.view(b * t, h_1, w_1, k)
        skip_x2 = x
        x = self.token_merge_2(x)
        h_2, w_2 = x.shape[1], x.shape[2]
        n_2 = h_2 * w_2
        x = x.view(b * t, n_2, k)
        x = x + self.spatial_pos_2.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_2:
            x, _ = layer(x, (b, t, n_2, k), cache_attn=cache_attn)
        
        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_2, w_2, k)
        skip_x2 = skip_x2.view(b * t, h_1, h_1, k)
        # Perform token splitting back to original number of patches
        x = self.token_split_2(x, skip_x2)  # x now has shape [b * t, skip_h_patches, skip_w_patches, k]
        # Reshape x back to [b * t, n_original, k]
        x = x.view(b * t, n_1, k)
        pos_1 = self.spatial_pos_1_dec.expand(b * t, -1, -1).type_as(x)
        x = x + pos_1
        # Spatiotemporal layers after splitting
        for layer in self.spacetime_layers_3:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn, local_only=True)

        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_1, w_1, k)
        skip_x1 = skip_x1.view(b * t, h_0, w_0, k)
        x = self.token_split_1(x, skip_x1)
        x = x.view(b * t, n, k)
        pos_0 = self.spatial_pos_0_dec.expand(b * t, -1, -1).type_as(x)
        x = x + pos_0
        for layer in self.spacetime_layers_4:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, local_only=True)
        
        # Run decoder layer
        x, _ = self.out_layer(x, x, cache_attn=cache_attn, local_only=True)
        x = self.out_activation(x)

        # Reshape and combine patches to form images
        x = x.reshape(b * t, n, c, p, p)
        x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
        x = x.view(b, t, c, h, w)

        return {'predictions': x}

    def reset_cache(self):
        for layer in self.spacetime_layers_0:
            layer.reset_cache()
        for layer in self.spacetime_layers_1:
            layer.reset_cache()
        for layer in self.spacetime_layers_2:
            layer.reset_cache()
        for layer in self.spacetime_layers_3:
            layer.reset_cache()
        for layer in self.spacetime_layers_4:
            layer.reset_cache()
        self.out_layer.reset_cache()


class PatchDecoderUA(nn.Module):
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
        self.spatial_pos_0 = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.spatial_pos_1 = nn.Parameter(torch.zeros(1, num_patches // 4, hidden_dim))
        self.spatial_pos_2 = nn.Parameter(torch.zeros(1, num_patches // 16, hidden_dim))

        # Token merge/split layers
        self.token_merge_1 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_merge_2 = TokenMerge(in_features=hidden_dim, out_features=hidden_dim, h_merge=2, w_merge=2)
        self.token_split_2 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)
        self.token_split_1 = TokenSplit(in_features=hidden_dim, out_features=hidden_dim, h_split=2, w_split=2)

        # Spatiotemporal attention layers
        self.spacetime_layers_0 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        # Spatiotemporal attention layers after merging
        self.spacetime_layers_1 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_2 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_3 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(4)])

        # Spatiotemporal attention layers
        self.spacetime_layers_4 = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
                           torch_attn=torch_attn) for _ in range(2)])

        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Final out layer
        self.out_layer = DecoderCrossAttentionBlock(hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, 
                                                    head_dim=head_dim, dropout=dropout, final_dropout=False, 
                                                    last_residual=False)
        
        self.out_activation = nn.Sigmoid()

    def forward(self, x, targets=None, cache_attn=False, seq_idx=None, cache_decoder_attn=False):
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
        x = x + self.spatial_pos_0.expand(b * t, -1, -1).type_as(x)
        
        # Spatiotemporal layers
        for layer in self.spacetime_layers_0:
            x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn)

        # Perform 2x2 token merge here. Input shape is [b * t, n, k]
        n = x.size(1)
        h_0 = w_0 = int(math.sqrt(n))

        # Reshape x to [b * t, h_patches, w_patches, k]
        x = x.view(b * t, h_0, w_0, k)
        # Apply token merging
        skip_x1 = x
        x = self.token_merge_1(x)  # x now has shape [b * t, h_patches // 2, w_patches // 2, k]
        # Update number of patches and reshape back to [b * t, n', k]
        h_1, w_1 = x.shape[1], x.shape[2]
        n_1 = h_1 * w_1
        x = x.view(b * t, n_1, k)
        x = x + self.spatial_pos_1.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_1:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn)

        x = x.view(b * t, h_1, w_1, k)
        skip_x2 = x
        x = self.token_merge_2(x)
        h_2, w_2 = x.shape[1], x.shape[2]
        n_2 = h_2 * w_2
        x = x.view(b * t, n_2, k)
        x = x + self.spatial_pos_2.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after merging
        for layer in self.spacetime_layers_3:
            x, _ = layer(x, (b, t, n_2, k), cache_attn=cache_attn)
        
        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_2, w_2, k)
        skip_x2 = skip_x2.view(b * t, h_1, h_1, k)
        # Perform token splitting back to original number of patches
        x = self.token_split_2(x, skip_x2)  # x now has shape [b * t, skip_h_patches, skip_w_patches, k]
        # Reshape x back to [b * t, n_original, k]
        x = x.view(b * t, n_1, k)
        x = x + self.spatial_pos_1.expand(b * t, -1, -1).type_as(x)

        # Spatiotemporal layers after splitting
        for layer in self.spacetime_layers_4:
            x, _ = layer(x, (b, t, n_1, k), cache_attn=cache_attn)

        # Perform token splitting back to original number of patches here
        x = x.view(b * t, h_1, w_1, k)
        skip_x1 = skip_x1.view(b * t, h_0, w_0, k)
        x = self.token_split_1(x, skip_x1)
        x = x.view(b * t, n, k)
        x = x + self.spatial_pos_0.expand(b * t, -1, -1).type_as(x)

        # Initialize list to collect outputs
        outputs = []

        self.out_layer.reset_cache()
        start = self.start_token.expand(b * t, -1, -1).type_as(x)
        memory = x.clone().detach()
        x_old = x

        # If targets, run training mode with teacher forcing
        if targets is not None:
            # Add random noise to half of the target patches at random
            targets = extract_tensor_patches(targets.view(b * t, c, h, w), p, p)
            targets = targets.view(b * t, n, c, p, p)
            targets = targets[:, :-1, ...]
            targets = add_noise_and_clip(targets, mean=0, std_range=(0, 0.5), mask_percent=0.5)
            
            # Embed target patches
            with torch.no_grad():
                targets = self.patch_embedder(targets.reshape(-1, c, p, p))
                targets = targets.view(b * t, n - 1, k)
                # Add spatial positional encoding
                targets = targets + self.spatial_pos.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]
            targets = torch.cat([start, targets], dim=1)
            
            x, _ = self.out_layer(x, memory, targets)
            x = self.out_activation(x)

        else:
            # Generate patches one by on
            for i in range(n):
                if i == 0:
                    target = start.clone()

                x = x_old[:, i, :].unsqueeze(1)
                x, _ = self.out_layer(x, memory, target, cache_attn=cache_decoder_attn)

                # Get the last output token
                x = self.out_activation(x)
                # Collect output
                outputs.append(x.clone())

                # Prepare next decoder input if not last patch
                if i < n - 1:
                    # Embed target patches using fixed patch embedder (do not update or learn)
                    with torch.no_grad():
                        new_target = self.patch_embedder(x.view(b * t, c, p, p))
                        new_target = new_target.reshape(b * t, 1, k)
                        pos_encoding = self.spatial_pos[:, i, :].type_as(new_target).unsqueeze(1)
                    new_target = new_target + pos_encoding  # Add positional encoding

                    if cache_decoder_attn:
                        target = new_target.clone().detach()
                    else:
                        target = torch.cat([target, new_target], dim=1).clone().detach()
                
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
        for layer in self.spacetime_layers_0:
            layer.reset_cache()
        for layer in self.spacetime_layers_1:
            layer.reset_cache()
        for layer in self.spacetime_layers_2:
            layer.reset_cache()
        for layer in self.spacetime_layers_3:
            layer.reset_cache()
        for layer in self.spacetime_layers_4:
            layer.reset_cache()
        self.out_layer.reset_cache()


class DecoderCrossAttentionBlockU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout

        # Pre-attention layer norm
        self.pre_norm = nn.RMSNorm(self.in_dim)

        # Self attention layer
        self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                             dropout=dropout)
        # Post-attention dropout and layernorm
        self.dropout_attn = nn.Dropout(dropout)
        self.post_norm = nn.RMSNorm(self.in_dim)

        # Cross attention layer
        self.cross_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                   dropout=dropout)
        # Post-cross-attention dropout and layernorm
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.post_cross_norm = nn.RMSNorm(self.in_dim)
        
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
        x = x + self.dropout_attn(self.attn(x, memory, memory, need_weights=False, is_causal=False,
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

        
class TransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, heads=1, head_dim=None, dropout=0, final_dropout=True,
                 last_residual=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.last_residual = last_residual
        self.final_dropout = final_dropout

        self.local_mask = None

        # Pre-attention layer norm
        self.pre_norm = nn.RMSNorm(self.in_dim)

        # Self attention layer
        self.attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                             dropout=dropout)
        # Post-attention dropout and layernorm
        self.dropout_attn = nn.Dropout(dropout)
        self.post_norm = nn.RMSNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, memory, cache_attn=False, local_only=False):
        attn_mask = None
        if local_only and self.local_mask is None:
            n = x.size(1)
            positions = torch.arange(n)

            # Compute grid dimensions
            H = W = int(math.sqrt(n))  # Ensure that n is a perfect square

            # Compute row and column indices
            row_indices = positions // W
            col_indices = positions % W

            # Calculate pairwise differences
            row_diff = torch.abs(row_indices.unsqueeze(1) - row_indices.unsqueeze(0))
            col_diff = torch.abs(col_indices.unsqueeze(1) - col_indices.unsqueeze(0))

            # Identify adjacency including diagonal patches
            adjacency = (row_diff <= 1) & (col_diff <= 1)  # Boolean matrix

            # Initialize the attention mask with zeros
            attn_mask = torch.zeros_like(adjacency, dtype=torch.float32)

            # Use masked_fill to set non-adjacent positions to -inf
            attn_mask = attn_mask.masked_fill(~adjacency, float('-inf'))

            # Ensure the attention mask is on the correct device and has the right data type
            attn_mask = attn_mask.to(x.device).type_as(x)
            self.local_mask = attn_mask

        elif local_only:
            attn_mask = self.local_mask
            
        # Pre-attention layer norm
        x = self.pre_norm(x)

        # Apply causal self attention
        x = x + self.dropout_attn(self.attn(x, x, x, need_weights=False, is_causal=False,
                                            cache_attn=cache_attn, attn_mask=attn_mask)[0])
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
        self.local_mask = None
        # Set head dimension default if not provided
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Pre-attention layer norm
        self.pre_norm = nn.RMSNorm(self.in_dim)

        # Spatial and temporal attention layers given attenton type
        self.spatial_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True, 
                                                     dropout=dropout)
        self.temporal_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                      dropout=dropout)
       
        # Post-attention dropout and layernorm
        self.dropout_space_attn = nn.Dropout(dropout)
        self.dropout_time_attn = nn.Dropout(dropout)
        self.post_norm = nn.RMSNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, size, padding_mask=None, return_attn=False, cache_attn=False, global_only=False, 
                local_only=False):            
        spatial_attn = None
        temporal_attn = None
        attn_mask = None
        b, t, n, k = size
    
        # Pre-attention layer norm
        x = self.pre_norm(x)

        if local_only and self.local_mask is None:
            positions = torch.arange(n)

            # Compute grid dimensions
            H = W = int(math.sqrt(n))  # Ensure that n is a perfect square

            # Compute row and column indices
            row_indices = positions // W
            col_indices = positions % W

            # Calculate pairwise differences
            row_diff = torch.abs(row_indices.unsqueeze(1) - row_indices.unsqueeze(0))
            col_diff = torch.abs(col_indices.unsqueeze(1) - col_indices.unsqueeze(0))

            # Identify adjacency including diagonal patches
            adjacency = (row_diff <= 1) & (col_diff <= 1)  # Boolean matrix

            # Initialize the attention mask with zeros
            attn_mask = torch.zeros_like(adjacency, dtype=torch.float32)

            # Use masked_fill to set non-adjacent positions to -inf
            attn_mask = attn_mask.masked_fill(~adjacency, float('-inf'))

            # Ensure the attention mask is on the correct device and has the right data type
            attn_mask = attn_mask.to(x.device).type_as(x)
            self.local_mask = attn_mask

        elif local_only:
            attn_mask = self.local_mask
            
        x = x + self.dropout_space_attn(self.spatial_attn(x, x, x, need_weights=False, is_causal=False, 
                                                          cache_attn=False, attn_mask=attn_mask)[0])
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
                                                              cache_attn=cache_attn, padding_mask=padding_mask,
                                                              local_causal=local_only)[0])
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
                padding_mask=None, local_causal=False):
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
            if is_causal and local_causal and query.size(1) > 1:
                tgt_len = query.size(1)
                src_len = key.size(1)
                device = query.device
                i_indices = torch.arange(tgt_len, device=device).unsqueeze(1)
                j_indices = torch.arange(src_len, device=device).unsqueeze(0)
                # Compute the difference
                diff = i_indices - j_indices
                # Allowed positions where diff is 0 or 1
                allowed_positions = (diff == 0) | (diff == 1)
                attn_mask = (~allowed_positions).float() * float('-inf')
                # convert nan values in attn_mask to 0
                attn_mask[attn_mask != attn_mask] = 0
                attn_mask = attn_mask.to(query.device).type_as(query)

            elif is_causal and query.size(1) > 1:
                tgt_len = query.size(1)
                src_len = key.size(1)
                attn_mask = torch.triu(torch.full((tgt_len, src_len), float('-inf'), device=query.device), diagonal=1)
            else:
                attn_mask = None
        # check if attn_mask has more than 2 dimensions
        elif len(attn_mask.size()) > 2:
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

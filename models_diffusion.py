'''Diffusion-based video prediction models'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers import DDPMScheduler
from einops import rearrange
from tools.model_tools import generate_spatial_mask, add_noise_and_clip
from kornia.contrib import extract_tensor_patches, combine_tensor_patches
from torch.nn.modules.transformer import _generate_square_subsequent_mask


def load_model_diff(args):
    """
    Model Loader for all the models contained within this file. Add new mappings here for external loading
    """
    model = None

    match args.model_name:
        case 'patch_decoder_diffusion':
            model = PatchDecoderDiff(**vars(args))
        case _:
            model = PatchDecoderDiff(**vars(args))

    return model


# class PatchDecoderDiff(nn.Module):
#     def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
#                  channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
#         super().__init__()
#         """
#         Uses auto-regressive patch decoding (patches predicted sequentially) with diffusion on the decoder side.
#         """

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
#         self.patch_embedder_dec = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

#         # Learnable encodings for temporal and spatial positions with random initialization
#         self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
#         self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
#         self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
#         self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
#         self.global_cache = None
        
#         # Spatiotemporal attention layers
#         self.spacetime_layers = nn.ModuleList([
#             SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
#                            torch_attn=torch_attn) for _ in range(layers)])

#         # Difussion decoder layers
#         self.diffusion_decoder = nn.ModuleList([DecoderCrossAttentionBlockGlobal(
#             hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
#             final_dropout=True, last_residual=True) for _ in range(decoding_layers)])

#         # Output diffusion layer
#         self.out_layer = DecoderCrossAttentionBlockGlobal(
#             hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout, 
#             final_dropout=False, last_residual=False)
        
#         # Output activation
#         self.out_activation = nn.Sigmoid()

#     def forward(self, x, padding_mask=None, targets=None, cache_attn=False, cache_decoder_attn=False, seq_idx=None):
#         b, t, c, h, w = x.size()  # Input batch dimensions: [batch, timesteps, channels, height, width]
#         n, p = self.num_patches, self.patch_size  # Number of patches and patch size

#         # Split into patches
#         x = extract_tensor_patches(x.view(b * t, c, h, w), p, p)

#         # If targets are provided, split targets into patches
#         if targets is not None:
#             # TODO add diffusion noise to target impages before patchifying
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
#         x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=t, k=k)

#         # Spatiotemporal layers
#         for layer in self.spacetime_layers:
#             x, _ = layer(x, (b, t, n, k), cache_attn=cache_attn, global_only=False)

#         # Ensure shape is temporally batched for patch decoding
#         x = x.view(b * t, n, k)

#         # remove global tokens from patches and store as x_global
#         x_global = x[:, 0, :].reshape(b, t, k)
#         x = x[:, 1:, :]
#         n = n - 1

#         if cache_attn and seq_idx is None:
#             self.global_cache = x_global
#         elif cache_attn and seq_idx is not None:
#             x_global = torch.cat([self.global_cache, x_global], dim=1)
#             self.global_cache = x_global

#         t_g = x_global.size(1)
#         # repeat t times in batch dimension
#         x_global = x_global.repeat_interleave(t, dim=0)
#         # Calculate attention mask for cross-attention to global tokens
#         attn_mask = torch.triu(torch.full((t_g, t), float('-inf'), device=x.device), diagonal=1)  # Shape: [t_g, t_g]
#         attn_mask = attn_mask.unsqueeze(1).expand(-1, n, -1)  # Shape: [t_g, n, t_g]
#         attn_mask = attn_mask.unsqueeze(0).expand(b, -1, -1, -1)  # Shape: [b, t_g, n, t_g]
#         attn_mask = attn_mask.reshape(b * t, n, t_g)
        
#         # If targets, run training mode with denoising targets
#         if targets is not None:
#             # Embed target patches
#             targets = self.patch_embedder_dec(targets.reshape(-1, c, p, p))
#             targets = targets.view(b * t, n - 1, k)
#             # Add spatial positional encoding
#             targets = targets + self.spatial_pos_dec.type_as(targets)[:, :-1, :]  # [b * t, n - 1, k]

#             for decoder_layer in self.patch_decoder:
#                 targets, _ = decoder_layer(targets, x_global, x, cross_attn_mask=attn_mask) 
#             # Project to output dimension
#             targets, _ = self.out_layer(targets, x_global, x, cross_attn_mask=attn_mask)
#             x = self.out_activation(targets)
            
#         # If no targets (eval), run autoregressive inference mode starting from noise input
#         else:
#             # TODO add inference for diffusion here

        
#         # Reshape and combine patches to form images
#         x = x.reshape(b * t, n, c, p, p)
#         x = combine_tensor_patches(x, original_size=h, window_size=p, stride=p)
#         x = x.view(b, t, c, h, w)

#         return {'predictions': x}

#     def reset_cache(self):
#         for layer in self.spacetime_layers:
#             layer.reset_cache()
#         for layer in self.patch_decoder:
#             layer.reset_cache()
#         self.out_layer.reset_cache()
#         self.global_cache = None


class PatchDecoderDiff(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_factor=1, layers=1, heads=1, head_dim=None, dropout=0,
                 channels=3, max_input_len=16, torch_attn=True, mc_drop=False, decoding_layers=1, **kwargs):
        super().__init__()
        """
        Uses auto-regressive patch decoding (patches predicted sequentially) with diffusion on the decoder side.
        """

        # Calculate patch and model dimensions
        patch_dim = patch_size ** 2
        num_patches = (img_size // patch_size) ** 2
        hidden_dim = patch_dim * embed_factor
        head_dim = hidden_dim if (head_dim == 0 or head_dim is None) else head_dim

        # Set model attributes
        self.patch_size = patch_size
        self.mc_drop = mc_drop
        self.num_patches = num_patches

        # Diffusion scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Linear patch encoder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)
        self.patch_embedder_dec = PatchEmbedder(patch_size=patch_size, projection_dim=hidden_dim, in_channels=channels)

        # Learnable encodings for temporal and spatial positions with random initialization
        self.temporal_pos = nn.Parameter(torch.randn(1, max_input_len, hidden_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.spatial_pos_dec = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)  # Adjusted size
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.global_cache = None

        # Spatiotemporal attention layers
        self.spacetime_layers = nn.ModuleList([
            SpaceTimeBlock(hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout,
                           torch_attn=torch_attn) for _ in range(layers)])

        # Diffusion decoder layers
        self.diffusion_decoder = nn.ModuleList([DecoderCrossAttentionBlockGlobal(
            hidden_dim, hidden_dim, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout,
            final_dropout=True, last_residual=True) for _ in range(decoding_layers)])

        # Output diffusion layer
        self.out_layer = DecoderCrossAttentionBlockGlobal(
            hidden_dim, patch_dim * channels, hidden_dim * 4, heads=heads, head_dim=head_dim, dropout=dropout,
            final_dropout=False, last_residual=False)

        # Output activation (removed for diffusion models)
        # self.out_activation = nn.Sigmoid()  # Typically not used in diffusion models

        # Time embedding for diffusion timesteps
        self.time_embed = SinusoidalEmbeddings(time_steps=1000, embed_dim=hidden_dim)

    def forward(self, x, t, steps=10, padding_mask=None, targets=None, cache_attn=False, cache_decoder_attn=False, 
                seq_idx=None):
        b, seq_len, c, h, w = x.size()  # [batch_size, timesteps, channels, height, width]
        n, p = self.num_patches, self.patch_size  # Number of patches and patch size

        # Flatten x and extract patches
        x = extract_tensor_patches(x.view(b * seq_len, c, h, w), p, p)

        # Patch embedding
        x = self.patch_embedder(x.view(b * seq_len * n, c, p, p)).view(b * seq_len, n, -1)
        k = x.size(-1)  # Hidden dimension

        # Add spatial positional encoding
        x = x + self.spatial_pos.type_as(x)

        # Add global token to patch dimension
        global_token = self.global_token.expand(b * seq_len, -1, -1).type_as(x)
        x = torch.cat([global_token, x], dim=1)
        n = n + 1  # Update number of patches including global token

        # Apply temporal positional encodings
        x = rearrange(x, '(b t) n k -> (b n) t k', b=b, n=n, t=seq_len, k=k)
        x = x + self.temporal_pos.type_as(x)[:, :seq_len, :]
        x = rearrange(x, '(b n) t k -> (b t) n k', b=b, n=n, t=seq_len, k=k)

        # Spatiotemporal layers
        for layer in self.spacetime_layers:
            x, _ = layer(x, (b, seq_len, n, k), cache_attn=cache_attn, global_only=False)

        # Reshape for decoder
        x = x.view(b * seq_len, n, k)

        # Separate global tokens
        x_global = x[:, 0, :].reshape(b, seq_len, k)
        x = x[:, 1:, :]
        n = n - 1  # Adjust number of patches

        if cache_attn and seq_idx is None:
            self.global_cache = x_global
        elif cache_attn and seq_idx is not None:
            x_global = torch.cat([self.global_cache, x_global], dim=1)
            self.global_cache = x_global

        t_g = x_global.size(1)
        # Repeat x_global for each timestep
        x_global = x_global.repeat_interleave(seq_len, dim=0)
        attn_mask = torch.triu(torch.full((t_g, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n, -1)
        attn_mask = attn_mask.unsqueeze(0).expand(b, -1, -1, -1)
        attn_mask = attn_mask.reshape(b * seq_len, n, t_g)

        # If targets are provided, training mode with denoising targets
        if targets is not None:
            # Get noise for targets
            targets = targets.reshape(-1, c, h, w)
            noise = torch.randn(targets.shape).to(targets.device)
            noisy_targets = self.scheduler.add_noise(targets, noise, t).to(targets.device)
            timestep_embedding = self.time_embed(noisy_targets, t)

            # Flatten targets and extract patches
            targets_patches = extract_tensor_patches(noisy_targets.view(b * seq_len, c, h, w), p, p)
            targets_patches = targets_patches.view(b * seq_len, n, c, p, p)

            # Embed target patches
            targets_embedded = self.patch_embedder_dec(targets_patches.view(-1, c, p, p))
            targets_embedded = targets_embedded.view(b * seq_len, n, k)
            
            # Add spatial positional encoding
            targets_embedded = targets_embedded + self.spatial_pos_dec.type_as(targets_embedded)

            # Decoder layers
            x_memory = x  # Encoder outputs
            # x_memory = torch.cat([timestep_embedding, x_memory], dim=1)
            x = targets_embedded
            # add timestep embedding to x

            for decoder_layer in self.diffusion_decoder:
                x = x + timestep_embedding
                x, _ = decoder_layer(x, x_global, x_memory, cache_attn=cache_decoder_attn,
                                     cross_attn_mask=attn_mask)

            # Output layer to predict noise
            x, _ = self.out_layer(x, x_global, x_memory, cross_attn_mask=attn_mask, cache_attn=cache_decoder_attn)
            
            x = x.view(b * seq_len, n, c, p, p)
            # Combine patches to reconstruct the predicted noise in image space
            predicted_noise_patches = x.view(b * seq_len, n, c, p, p)
            predicted_noise = combine_tensor_patches(predicted_noise_patches, original_size=h, window_size=p, stride=p)
            predicted_noise = predicted_noise.view(b, seq_len, c, h, w)  # Shape: [b, seq_len, c, h, w]
            return {'predictions': predicted_noise, 'target_noise': noise}

        # If no targets (inference mode), run sampling (iterative denoising)
        else:
            # Inference mode: iterative denoising
            timesteps = self.scheduler.set_timesteps(50)
            noise = torch.randn(b * seq_len, c, h, w, device=x.device)
            
            # Start the denoising loop
            for t_step in self.scheduler.timesteps:
                # scale noise between -1 and 1
                timestep_embedding = self.time_embed(noise, t_step.expand(b * seq_len))

                # Flatten targets and extract patches
                targets_patches = extract_tensor_patches(noise.view(b * seq_len, c, h, w), p, p)
                targets_patches = targets_patches.view(b * seq_len, n, c, p, p)

                # Embed target patches
                targets_embedded = self.patch_embedder_dec(targets_patches.view(-1, c, p, p))
                targets_embedded = targets_embedded.view(b * seq_len, n, k)
                
                # Add spatial positional encoding
                targets_embedded = targets_embedded + self.spatial_pos_dec.type_as(targets_embedded)

                # Decoder layers
                x_memory = x  # Encoder outputs
                # x_memory = torch.cat([timestep_embedding, x_memory], dim=1)
                x_new = targets_embedded
                # add timestep embedding to x

                for decoder_layer in self.diffusion_decoder:
                    x_new = x_new + timestep_embedding
                    x_new, _ = decoder_layer(x_new, x_global, x_memory, cache_attn=cache_decoder_attn,
                                             cross_attn_mask=attn_mask)

                # Output layer to predict noise
                x_new, _ = self.out_layer(x_new, x_global, x_memory, cross_attn_mask=attn_mask, 
                                          cache_attn=cache_decoder_attn)
                
                x_new = x_new.view(b * seq_len, n, c, p, p)
                # Combine patches to reconstruct the predicted noise in image space
                predicted_noise_patches = x_new.view(b * seq_len, n, c, p, p)
                predicted_noise = combine_tensor_patches(predicted_noise_patches, original_size=h, window_size=p, stride=p)
                predicted_noise = predicted_noise.view(b * seq_len, c, h, w)  # Shape: [b, seq_len, c, h, w]

                previous_noisy_sample = self.scheduler.step(predicted_noise, t_step, noise).prev_sample
                noise = previous_noisy_sample

            predicted = (noise / 2 + 0.5).clamp(0, 1)
            return {'predictions': predicted.reshape(b, seq_len, c, h, w)}

    def reset_cache(self):
        for layer in self.spacetime_layers:
            layer.reset_cache()
        for layer in self.diffusion_decoder:
            layer.reset_cache()
        self.out_layer.reset_cache()
        self.global_cache = None


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

    
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings.to('cuda')

    def forward(self, x, t):
        embeds = self.embeddings[t]
        return embeds[:, :].unsqueeze(1)
        

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

        self.self_attn = CachedMultiheadAttention(embed_dim=self.in_dim, num_heads=heads, batch_first=True,
                                                  dropout=dropout)

        self.dropout_self_attn = nn.Dropout(dropout)
        self.post_self_norm = nn.LayerNorm(self.in_dim)
        
        # Feedforward layer
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout_fc = nn.Dropout(dropout) if final_dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

        # self.out_block = TransformerBlock(self.in_dim, self.out_dim, self.out_dim, heads=heads, head_dim=head_dim,
        #                                   dropout=dropout, final_dropout=final_dropout, last_residual=last_residual)

    def forward(self, targets, global_x, x, padding_mask=None, cache_attn=False, cross_attn_mask=None):
        # Pre-attention layer norm
        # x = self.pre_norm(x)

        x = x + self.dropout_self_attn(self.self_attn(x, x, x, need_weights=False, is_causal=False, 
                                                cache_attn=cache_attn)[0])
        x = self.post_self_norm(x)

        # Apply causal self attention
        x = x + self.dropout_attn(self.attn(targets, global_x, global_x, need_weights=False, is_causal=True,
                                            padding_mask=padding_mask, cache_attn=False, attn_mask=cross_attn_mask)[0])
        x = self.post_norm(x)

        # Apply cross attention
        x = x + self.dropout_cross_attn(self.cross_attn(targets, x, x, need_weights=False, is_causal=False,
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

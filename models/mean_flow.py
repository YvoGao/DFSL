"""
the flow matching network, from MAR.
Simplified version: No Disentanglement, with Mean Flow Inference.
"""

import math
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * 1000.0
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block with adaptive layer norm.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DeepFlowMatchingNet(nn.Module):
    """
    Standard Flow Matching Network for Velocity Field.
    Supports both Multi-Step Euler inference and Mean Flow single-step inference.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding
        self.time_embed = TimestepEmbedder(model_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Residual blocks
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        # Final layer
        self.final_layer = FinalLayer(model_channels, out_channels)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Forward pass for training (predicting velocity at any timestep t).
        :param x: an [N x C] Tensor of interpolated features.
        :param t: a 1-D batch of timesteps.
        :return: an [N x C] Tensor of velocity.
        """
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        y = t_emb

        for block in self.res_blocks:
            x = block(x, y)
        
        velocity = self.final_layer(x, y)
        return velocity

    def mean_flow_inference(self, x0):
        """
        Mean Flows / Rectified Flow 单步推理 (Straight Flow).
        核心逻辑：假设速度场恒定，直接积分 x1 = x0 + v(x0, t=0).
        这是 "Training-free" 的单步推理，不需要修改训练流程。
        
        :param x0: 初始图像特征 [N x C]
        :return: 对齐后的特征 [N x C]
        """
        self.eval()
        with torch.no_grad():
            # 1. 设置 t=0 (起始点)
            t = torch.zeros((x0.shape[0], 1), device=x0.device)
            
            # 2. 预测 t=0 时刻的速度
            # 在 Rectified Flow 中，v(x, 0) 通常指向 x1 的方向
            velocity_pred = self.forward(x0, t)
            
            # 3. 单步积分：x1 = x0 + 1.0 * velocity
            # 因为总时间 T=1，步长 h=1
            x1 = x0 + velocity_pred
            
        return x1
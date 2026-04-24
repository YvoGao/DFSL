"""
Bidirectional Flow Matching Network: Image ← Shared Distribution → Text
两个独立速度场，分别负责图像→共享分布、文本→共享分布
"""

import math
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
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


class BidirectionalFlowNet(nn.Module):
    """
    双向流核心模型：
    - flow_img: 图像特征 → 共享中间分布 的速度场
    - flow_text: 文本特征 → 共享中间分布 的速度场
    两个流独立权重，可分别适配图像/文本的分布特性
    """
    def __init__(
        self,
        in_channels,    # CLIP特征维度，图像和文本维度一致
        model_channels,
        num_res_blocks,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels

        # 共享时间嵌入层（两个流共用，减少参数量）
        self.time_embed = TimestepEmbedder(model_channels)

        # ------------------- 图像流：图像→共享分布 -------------------
        self.img_input_proj = nn.Linear(in_channels, model_channels)
        self.img_res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])
        self.img_final_layer = FinalLayer(model_channels, in_channels)

        # ------------------- 文本流：文本→共享分布 -------------------
        self.text_input_proj = nn.Linear(in_channels, model_channels)
        self.text_res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])
        self.text_final_layer = FinalLayer(model_channels, in_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化时间嵌入
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out 输出层，保证训练初始稳定
        for block in self.img_res_blocks + self.text_res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for final_layer in [self.img_final_layer, self.text_final_layer]:
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

    def forward_img_flow(self, x, t):
        """图像流前向：预测图像→共享分布的速度"""
        x = self.img_input_proj(x)
        t_emb = self.time_embed(t)
        y = t_emb
        for block in self.img_res_blocks:
            x = block(x, y)
        velocity = self.img_final_layer(x, y)
        return velocity

    def forward_text_flow(self, z, t):
        """文本流前向：预测文本→共享分布的速度"""
        z = self.text_input_proj(z)
        t_emb = self.time_embed(t)
        y = t_emb
        for block in self.text_res_blocks:
            z = block(z, y)
        velocity = self.text_final_layer(z, y)
        return velocity

    def forward(self, x, t, mode="img"):
        """统一前向接口，mode区分图像流/文本流"""
        if mode == "img":
            return self.forward_img_flow(x, t)
        elif mode == "text":
            return self.forward_text_flow(x, t)
        else:
            raise ValueError(f"mode must be 'img' or 'text', got {mode}")

    # ------------------- 推理接口：单步映射到共享分布 -------------------
    def img_to_shared(self, x0, tau=1.0):
        """单步推理：图像特征 → 共享分布"""
        self.eval()
        with torch.no_grad():
            t = torch.zeros((x0.shape[0], 1), device=x0.device)
            v = self.forward_img_flow(x0, t)
            x_shared = x0 + tau * v
        return x_shared

    def text_to_shared(self, z0, tau=1.0):
        """单步推理：文本特征 → 共享分布"""
        self.eval()
        with torch.no_grad():
            t = torch.zeros((z0.shape[0], 1), device=z0.device)
            v = self.forward_text_flow(z0, t)
            z_shared = z0 + tau * v
        return z_shared
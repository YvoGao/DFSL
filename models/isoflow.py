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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
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

class IsoFlowNet(nn.Module):
    """
    各向同性子空间流匹配模型（轻量、稳定、不崩）
    """
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks):
        super().__init__()
        self.time_embed = TimestepEmbedder(model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        y = t_emb
        for block in self.res_blocks:
            x = block(x, y)
        velocity = self.final_layer(x, y)
        return velocity

    @torch.no_grad()
    def single_step_inference(self, x0, class_embeddings):
        device = x0.device
        B = x0.shape[0]

        t_zero = torch.zeros(B, 1, device=device)
        v = self.forward(x0, t_zero)

        # 自适应步长（子空间内更稳定）
        x0_n = x0 / x0.norm(dim=-1, keepdim=True)
        cls_n = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        sim = (x0_n @ cls_n.T).softmax(dim=-1)
        conf, _ = sim.max(dim=-1)

        base_step = 0.1
        scale = 1.0 - conf
        step = base_step * (1 + scale.unsqueeze(-1))

        x_aligned = x0 + step * v
        return x_aligned
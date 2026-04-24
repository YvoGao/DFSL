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

class DeepFlowMatchingNet(nn.Module):
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

    def mean_flow_inference(self, x0, class_embeddings):
        """
        创新点2：置信度引导的自适应单步推理
        输入：原始图像特征x0、全量文本特征class_embeddings
        输出：自适应对齐后的特征
        """
        self.eval()
        with torch.no_grad():
            device = x0.device
            batch_size = x0.shape[0]
            t_zero = torch.zeros(batch_size, 1, device=device)
            
            # 1. 预测t=0时的速度（和训练自校准路径完全一致）
            velocity_pred = self.forward(x0, t_zero)
            
            # 2. 自适应步长计算：用初始置信度引导步长
            x0_norm = x0 / x0.norm(dim=-1, keepdim=True)
            class_norm = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            init_sim = (x0_norm @ class_norm.T).softmax(dim=-1)
            max_conf, _ = init_sim.max(dim=-1)
            
            # 置信度低的难样本步长大，置信度高的简单样本步长小，避免过修正
            base_step = 0.15
            step_scale = 1.0 - max_conf  # 置信度越低，缩放系数越大
            adaptive_step = base_step * (1 + step_scale.unsqueeze(-1))
            
            # 3. 自适应单步对齐
            x_aligned = x0 + adaptive_step * velocity_pred
            
        return x_aligned
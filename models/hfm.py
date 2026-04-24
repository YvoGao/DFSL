import torch
import torch.nn as nn
from .hyperbolic import poincare_to_euclidean, euclidean_to_poincare, poincare_distance, HyperbolicProjection

class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, t):
        # t: (B,1)
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(self.max_period)) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.time_emb = nn.Linear(dim, dim)
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.linear1(h)
        h = h + self.time_emb(t_emb)
        h = self.act(h)
        h = self.norm2(h)
        h = self.linear2(h)
        return x + h

class HFMNet(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, num_blocks=3, c=1.0):
        super().__init__()
        self.c = c
        self.time_emb = TimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, dim)
        # 双曲投影
        self.hyper_proj = HyperbolicProjection(dim, c=c)
    def forward(self, z, t):
        # z: (B,d) 双曲特征
        # t: (B,1) 时间
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)
        # 双曲 -> 欧式
        x = poincare_to_euclidean(z, self.c)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        # 预测欧式速度
        v_euc = self.output_proj(h)
        # 转回双曲速度（简化版）
        return v_euc

    def get_loss(self, img_feat, txt_feat):
        # img_feat, txt_feat: (B,d) 欧式CLIP特征
        B = img_feat.size(0)
        device = img_feat.device
        # 1. 投影到双曲空间
        z0 = self.hyper_proj(img_feat)   # 图像特征（源）
        z1 = self.hyper_proj(txt_feat)   # 文本特征（目标）
        # 2. 时间采样
        t = torch.rand(B, 1, device=device)
        # 3. 双曲插值（简化：欧式插值再投影）
        z_t_euc = (1 - t) * poincare_to_euclidean(z0, self.c) + t * poincare_to_euclidean(z1, self.c)
        z_t = euclidean_to_poincare(z_t_euc, self.c)
        # 4. 预测速度
        v_pred_euc = self(z_t, t)
        # 5. 真值速度（欧式）
        v_gt_euc = poincare_to_euclidean(z1, self.c) - poincare_to_euclidean(z0, self.c)
        # 6. 流匹配损失 + 双曲距离损失
        loss_fm = torch.mean((v_pred_euc - v_gt_euc) ** 2)
        loss_dist = torch.mean(poincare_distance(z0, z1, self.c))
        return loss_fm + 0.01 * loss_dist

    @torch.no_grad()
    def inference(self, img_feat):
        # 单步推理
        z0 = self.hyper_proj(img_feat)
        t0 = torch.zeros(img_feat.size(0), 1, device=img_feat.device)
        v_euc = self(z0, t0)
        # 单步更新
        z0_euc = poincare_to_euclidean(z0, self.c)
        z_aligned_euc = z0_euc + 0.1 * v_euc
        z_aligned = euclidean_to_poincare(z_aligned_euc, self.c)
        # 转回欧式用于分类
        aligned_feat = poincare_to_euclidean(z_aligned, self.c)
        return aligned_feat
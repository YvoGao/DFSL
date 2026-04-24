import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        h = self.in_ln(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return x + gate_mlp * h

# class DCMDNet(nn.Module):
#     """
#     完全对齐官网DCMD：梯度引导漂移 + 条件矩匹配 + 小样本适配
#     """
#     def __init__(self, dim=512, hidden_dim=512, num_blocks=3):
#         super().__init__()
#         # 移除官网无的TimestepEmbedder（冗余），用固定调制向量兼容ResBlock
#         self.ada_emb = nn.Parameter(torch.randn(hidden_dim))  
#         self.input_proj = nn.Linear(dim, hidden_dim)
#         self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
#         self.final_layer = nn.Linear(hidden_dim, dim)
        
#         # 初始化：漂移量初始为0（不破坏CLIP原生对齐，和官网一致）
#         nn.init.zeros_(self.final_layer.weight)
#         nn.init.zeros_(self.final_layer.bias)

#     def forward(self, x):
#         """
#         前向预测漂移位移（无时间步，纯DCMD逻辑）
#         """
#         # 固定调制向量（扩展到batch维度）
#         y = self.ada_emb.unsqueeze(0).repeat(x.shape[0], 1)  
#         h = self.input_proj(x)
#         for block in self.blocks:
#             h = block(h, y)
#         drift = self.final_layer(h)
#         return drift

#     def get_drifting_loss(self, img_feat, txt_feat, labels, temperature=100.0, eta=0.05):
#         """
#         完整复刻官网DCMD损失：MSE（漂移目标） + 0.1×条件矩匹配损失
#         """
#         B, D = img_feat.shape
#         C = txt_feat.shape[0]
#         device = img_feat.device

#         # 安全兜底（官网风格）
#         if labels.max() >= C or labels.min() < 0:
#             labels = torch.clamp(labels, 0, C-1)

#         # 开启特征梯度（精简逻辑，和官网jax对齐）
#         img_feat = img_feat.requires_grad_(True)

#         # 计算log p(y|x)（归一化+温度缩放，官网核心）
#         img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
#         txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
#         logits = img_norm @ txt_norm.T * temperature
#         log_p = F.log_softmax(logits, dim=-1)
#         log_p_y = log_p[torch.arange(B), labels]

#         # 梯度计算（构造漂移目标）
#         grads = torch.autograd.grad(
#             outputs=log_p_y.sum(),
#             inputs=img_feat,
#             create_graph=True,
#             retain_graph=True,
#         )[0]
#         target = img_feat + eta * grads

#         # 预测漂移并计算核心MSE损失
#         pred_drift = self.forward(img_feat.detach())
#         pred = img_feat.detach() + pred_drift
#         loss_mse = F.mse_loss(pred, target.detach())

#         # 补充官网的条件矩匹配损失（类别维度均值匹配，λ=0.1）
#         loss_moment = 0.0
#         for c in range(C):
#             mask = labels == c
#             if mask.sum() == 0: continue
#             img_mean = img_feat[mask].mean(dim=0)
#             loss_moment += F.mse_loss(img_mean, txt_feat[c])
#         loss_moment = loss_moment / max(C, 1)  # 空类别兜底

#         # 官网总损失（核心+矩匹配）
#         loss = loss_mse + 0.1 * loss_moment

#         return loss

#     @torch.no_grad()
#     def inference(self, img_feat):
#         """
#         推理逻辑和官网一致：单步漂移叠加
#         """
#         drift = self.forward(img_feat)
#         return img_feat + 0.1 * drift  # 缩放系数和官网默认一致

# 你的初始DCMDNet（保留，只改损失）
class DCMDNet(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, num_blocks=3):
        super().__init__()
        # 【保留】你的初始时间步嵌入（虽然官网没有，但它有用）
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        # 【保留】固定t=0，兼容你的初始逻辑
        t = torch.zeros(x.shape[0], 1, device=x.device)
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        drift = self.final_layer(h)
        return drift

    # 【核心修改】只优化损失：保留你的初始梯度引导，**极弱**加官网矩匹配
    def get_drifting_loss(self, img_feat, txt_feat, labels, temperature=100.0, eta=0.05, lambda_moment=0.01):
        B, D = img_feat.shape
        C = txt_feat.shape[0]
        device = img_feat.device

        if labels.max() >= C or labels.min() < 0:
            labels = torch.clamp(labels, 0, C-1)

        img_feat = img_feat.detach().requires_grad_(True)

        # 【保留】你的初始核心：梯度引导漂移
        img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = img_norm @ txt_norm.T * temperature
        log_p = F.log_softmax(logits, dim=-1)
        log_p_y = log_p[torch.arange(B), labels]
        grads = torch.autograd.grad(
            outputs=log_p_y.sum(),
            inputs=img_feat,
            create_graph=True,
            retain_graph=True,
        )[0]
        target = img_feat + eta * grads
        pred_drift = self.forward(img_feat.detach())
        pred = img_feat.detach() + pred_drift
        loss_mse = F.mse_loss(pred, target.detach())

        # 【新增】极弱的官网矩匹配（λ=0.01，几乎不干扰核心损失）
        loss_moment = 0.0
        for c in range(C):
            mask = labels == c
            if mask.sum() == 0: continue
            img_mean = img_feat[mask].mean(dim=0)
            loss_moment += F.mse_loss(img_mean, txt_feat[c])
        loss_moment = loss_moment / max(C, 1)

        # 总损失：核心MSE为主，矩匹配为辅
        total_loss = loss_mse + lambda_moment * loss_moment
        return total_loss
    
    @torch.no_grad()
    def inference(self, img_feat):
        """
        推理逻辑和官网一致：单步漂移叠加
        """
        drift = self.forward(img_feat)
        return img_feat + 0.1 * drift  # 缩放系数和官网默认一致
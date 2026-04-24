import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================
# 保留你原始稳定版的基础模块（不改动，保证兼容性）
# ======================
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


# ======================
# 优化后的DCMD核心模型（兼容你原有调用逻辑）
# ======================
class DCMDNet(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, num_blocks=3):
        super().__init__()
        # 保留你原始的时间步嵌入（固定t=0，保证训练稳定）
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, dim)
        
        # 初始化：输出层权重为0，不破坏CLIP原生对齐
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        """
        前向传播：预测漂移位移（和你原始接口完全一致）
        :param x: 原始图像特征 [B, D]
        :return: 漂移位移 [B, D]
        """
        # 固定t=0，和你原始逻辑完全一致
        t = torch.zeros(x.shape[0], 1, device=x.device)
        t_emb = self.time_embed(t)
        
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        drift = self.final_layer(h)
        return drift

    def get_drifting_loss(self, img_feat, txt_feat, labels, decouple_mask=None, temperature=100.0, enable_dynamic_step=True):
        """
        优化后的漂移损失函数（核心改进）
        :param img_feat: 图像特征 [B, D]
        :param txt_feat: 全局类别文本原型 [C, D]
        :param labels: 样本标签 [B]
        :param decouple_mask: 特征解耦掩码 [1, D]（APE风格，None则关闭解耦，回到原始版）
        :param temperature: 分类温度系数
        :param enable_dynamic_step: 是否开启动态步长（False则固定eta=0.05，回到原始版）
        :return: 标量损失
        """
        B, D = img_feat.shape
        C = txt_feat.shape[0]
        device = img_feat.device

        # 1. 安全兜底（防止标签越界）
        if labels.max() >= C or labels.min() < 0:
            labels = torch.clamp(labels, 0, C-1)

        # 2. 开启图像特征梯度（和原始逻辑一致）
        img_feat = img_feat.detach().requires_grad_(True)

        # 3. 特征归一化
        img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        # 4. 【核心改进1】APE风格特征解耦：仅用语义通道计算相似度
        if decouple_mask is not None:
            img_norm = img_norm * decouple_mask
            txt_norm = txt_norm * decouple_mask

        # 5. 计算分类概率与梯度
        logits = img_norm @ txt_norm.T * temperature
        log_p = F.log_softmax(logits, dim=-1)
        log_p_y = log_p[torch.arange(B), labels]

        # 6. 梯度计算
        grads = torch.autograd.grad(
            outputs=log_p_y.sum(),
            inputs=img_feat,
            create_graph=True,
            retain_graph=True,
        )[0]

        # 解耦掩码过滤梯度：仅语义通道保留梯度
        if decouple_mask is not None:
            grads = grads * decouple_mask
        grads = grads / (grads.norm(dim=-1, keepdim=True) + 1e-8)

        # 7. 【核心改进2】动态步长（根据样本难度自适应）
        if enable_dynamic_step:
            # 计算每个样本与对应类别原型的相似度，判断难度
            sample_sim = (img_norm * txt_norm[labels]).sum(dim=-1)
            # 简单样本（高相似度）小步长，难样本（低相似度）大步长
            eta = torch.where(
                sample_sim > 0.7, torch.tensor(0.01, device=device),
                torch.where(sample_sim < 0.3, torch.tensor(0.1, device=device),
                            torch.tensor(0.05, device=device))
            )
            eta = eta.unsqueeze(-1)
        else:
            # 固定步长，回到你原始版逻辑
            eta = 0.05

        # 8. 构造漂移目标
        target = img_feat + eta * grads

        # 9. 预测漂移
        pred_drift = self.forward(img_feat.detach())
        # 解耦掩码过滤漂移：仅语义通道有漂移，冗余通道完全保留原始CLIP特征
        if decouple_mask is not None:
            pred_drift = pred_drift * decouple_mask

        # 动态缩放推理步长，和训练保持一致
        if enable_dynamic_step:
            pred_scale = eta.detach() * 2
        else:
            pred_scale = 0.1
        pred = img_feat.detach() + pred_drift * pred_scale

        # 10. 损失计算：仅语义通道计算MSE，彻底排除冗余维度干扰
        if decouple_mask is not None:
            loss = F.mse_loss(pred * decouple_mask, target.detach() * decouple_mask)
        else:
            loss = F.mse_loss(pred, target.detach())

        return loss

    @torch.no_grad()
    def inference(self, img_feat, txt_feat, decouple_mask=None, enable_dynamic_step=True):
        """
        推理函数（和训练逻辑完全对齐，兼容你原有调用）
        :param img_feat: 图像特征 [B, D]
        :param txt_feat: 全局类别文本原型 [C, D]
        :param decouple_mask: 特征解耦掩码（和训练一致）
        :param enable_dynamic_step: 是否开启动态步长（和训练一致）
        :return: 漂移对齐后的特征 [B, D]
        """
        # 1. 预测漂移量
        drift = self.forward(img_feat)
        # 解耦掩码过滤
        if decouple_mask is not None:
            drift = drift * decouple_mask

        # 2. 动态步长缩放（和训练逻辑对齐）
        if enable_dynamic_step:
            img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
            txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
            if decouple_mask is not None:
                img_norm = img_norm * decouple_mask
                txt_norm = txt_norm * decouple_mask
            # 计算每个样本与最近原型的相似度
            sim = (img_norm.unsqueeze(1) * txt_norm.unsqueeze(0)).sum(dim=-1).max(dim=-1)[0]
            # 自适应步长
            pred_scale = torch.where(
                sim > 0.7, torch.tensor(0.02, device=img_feat.device),
                torch.where(sim < 0.3, torch.tensor(0.2, device=img_feat.device),
                            torch.tensor(0.1, device=img_feat.device))
            )
            pred_scale = pred_scale.unsqueeze(-1)
        else:
            # 固定步长，回到原始版
            pred_scale = 0.1

        # 3. 漂移对齐
        return img_feat + drift * pred_scale
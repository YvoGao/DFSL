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

class DCMDNet(nn.Module):
    """
    判别式跨模态漂移对齐模型
    完全复用FMA的模型结构，仅修改前向和损失函数
    """
    def __init__(self, dim=512, hidden_dim=512, num_blocks=3):
        super().__init__()
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, dim)
        
        # 初始化：输出层权重为0，训练初始时漂移量为0，不破坏CLIP原生对齐
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        """
        前向传播：预测漂移位移
        :param x: 原始图像特征 [B, D]
        :return: 漂移位移 [B, D]
        """
        # 漂移模型不需要时间步！这是和流匹配最大的区别
        # 我们用固定的t=0嵌入，保持模型结构和FMA兼容，方便复用代码
        t = torch.zeros(x.shape[0], 1, device=x.device)
        t_emb = self.time_embed(t)
        
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        drift = self.final_layer(h)
        return drift

    # def get_drifting_loss(self, img_feat, txt_feat, labels, temperature=100.0, eta=0.05):
    #     """
    #     何恺明 Drifting Model 正确 + 安全 + 适配 FSL 小样本
    #     :param img_feat: [B, D]
    #     :param txt_feat: [C, D]  小样本episode的类别原型
    #     :param labels:   [B]     0 ~ C-1
    #     """
    #     B, D = img_feat.shape
    #     C = txt_feat.shape[0]
    #     device = img_feat.device

    #     # ------------------------------
    #     # 安全检查（防止越界，必加）
    #     # ------------------------------
    #     if labels.max() >= C or labels.min() < 0:
    #         labels = torch.clamp(labels, 0, C-1)

    #     # ------------------------------
    #     # 1. 开启图像特征梯度
    #     # ------------------------------
    #     img_feat = img_feat.detach().requires_grad_(True)

    #     # ------------------------------
    #     # 2. 计算 log p(y|x)
    #     # ------------------------------
    #     img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
    #     txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

    #     logits = img_norm @ txt_norm.T  # [B, C]
    #     logits = logits * temperature

    #     # 取当前样本对应类别的 log 概率
    #     log_p = F.log_softmax(logits, dim=-1)
    #     log_p_y = log_p[torch.arange(B), labels]  # [B]

    #     # ------------------------------
    #     # 3. 计算梯度 → drift target
    #     # ------------------------------
    #     grads = torch.autograd.grad(
    #         outputs=log_p_y.sum(),
    #         inputs=img_feat,
    #         create_graph=True,
    #         retain_graph=True,
    #     )[0]

    #     target = img_feat + eta * grads

    #     # ------------------------------
    #     # 4. 网络预测漂移
    #     # ------------------------------
    #     pred_drift = self.forward(img_feat.detach())
    #     pred = img_feat.detach() + pred_drift

    #     # ------------------------------
    #     # 5. 损失
    #     # ------------------------------
    #     loss = F.mse_loss(pred, target.detach())

    #     return loss
    
    def get_drifting_loss(self, img_feat, txt_feat, labels, temperature=100.0):
        B, D = img_feat.shape
        C = txt_feat.shape[0]
        device = img_feat.device

        # 1. 安全检查+特征归一化（保留）
        if labels.max() >= C or labels.min() < 0:
            labels = torch.clamp(labels, 0, C-1)
        img_feat = img_feat.detach().requires_grad_(True)
        img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        # 2. 计算img-txt相似度（判断样本难度）
        sim = (img_norm.unsqueeze(1) * txt_norm.unsqueeze(0)).sum(dim=-1)  # [B, C]
        sample_sim = sim[torch.arange(B), labels]  # [B]：每个样本与同类文本的相似度

        # 3. 动态eta（根据相似度调整漂移步长）
        # 简单样本（sim>0.7）：eta=0.01（小步）；难样本（sim<0.3）：eta=0.1（大步）
        eta = torch.where(sample_sim > 0.7, torch.tensor(0.01, device=device),
                        torch.where(sample_sim < 0.3, torch.tensor(0.1, device=device),
                                    torch.tensor(0.05, device=device)))
        eta = eta.unsqueeze(-1)  # [B, 1]：每个样本专属eta

        # 4. 梯度引导（保留）
        logits = img_norm @ txt_norm.T * temperature
        log_p = F.log_softmax(logits, dim=-1)
        log_p_y = log_p[torch.arange(B), labels]
        grads = torch.autograd.grad(outputs=log_p_y.sum(), inputs=img_feat, create_graph=True, retain_graph=True)[0]
        target = img_feat + eta * grads  # 动态目标

        # 5. 预测漂移（保留）
        pred_drift = self.forward(img_feat.detach())
        
        # 6. 动态推理缩放（与训练eta保持一致，避免训练-推理偏差）
        pred_scale = eta.detach() * 2  # 缩放系数与eta正相关
        pred = img_feat.detach() + pred_drift * pred_scale

        # 7. 难度分类损失（重点：对难样本加大损失权重）
        loss_weight = torch.where(sample_sim < 0.3, torch.tensor(2.0, device=device), torch.tensor(1.0, device=device))
        loss = (loss_weight.unsqueeze(-1) * (pred - target.detach()) ** 2).mean()

        return loss

    @torch.no_grad()
    def inference(self, img_feat):
        """
        单步推理：直接加漂移位移
        """
        drift = self.forward(img_feat)
        return img_feat + 0.1 * drift  # 0.1是基础缩放系数，可微调
    
    # 在DCMDNet中新增
    def get_drifting_loss_mixup(self, mixed_img_feat, mixed_txt_feat, lam, temperature=100.0, eta=0.05):
        B, D = mixed_img_feat.shape
        
        mixed_img_feat = mixed_img_feat.detach().requires_grad_(True)

        # 归一化
        img_norm = mixed_img_feat / (mixed_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = mixed_txt_feat / (mixed_txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        # 一对一图文匹配（无类别索引，更稳定）
        logits = (img_norm * txt_norm).sum(dim=-1, keepdim=True) * temperature
        log_p = F.logsigmoid(logits)

        # 计算梯度
        grads = torch.autograd.grad(
            outputs=log_p.sum(),
            inputs=mixed_img_feat,
            create_graph=True,
            retain_graph=True,
        )[0]
        target = mixed_img_feat + eta * grads

        # 预测与损失
        pred_drift = self.forward(mixed_img_feat.detach())
        pred = mixed_img_feat.detach() + pred_drift
        loss = F.mse_loss(pred, target.detach())
        return loss
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

class LocalResBlock(nn.Module):
    """轻量类别级局部残差块"""
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

class DGFM(nn.Module):
    """
    判别式引导的小样本流匹配（DGFM）核心模型
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        num_classes,        # 数据集类别数
        num_res_blocks=2,    # 每个类别的局部残差块数量（轻量）
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_classes = num_classes

        # 共享时间嵌入层
        self.time_embed = TimestepEmbedder(model_channels)

        # 共享输入投影层
        self.shared_proj = nn.Linear(in_channels, model_channels)

        # 类别级局部流场：每个类别对应独立的轻量残差块
        self.class_flow_blocks = nn.ModuleDict({
            f"class_{i}": nn.ModuleList([LocalResBlock(model_channels) for _ in range(num_res_blocks)])
            for i in range(num_classes)
        })

        # 共享输出层
        self.final_layer = FinalLayer(model_channels, in_channels)

        # 动态文本锚点：移动平均更新
        self.register_buffer("text_anchors", torch.zeros(num_classes, in_channels))
        self.anchor_momentum = 0.99

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

        # Zero-out输出层，保证训练初始稳定
        for class_blocks in self.class_flow_blocks.values():
            for block in class_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def update_text_anchors(self, text_features, labels):
        """动态更新文本锚点（训练时调用）"""
        with torch.no_grad():
            for i, label in enumerate(labels):
                self.text_anchors[label] = self.anchor_momentum * self.text_anchors[label] + (1 - self.anchor_momentum) * text_features[i]

    def forward(self, x, t, labels=None):
        """
        前向传播
        :param x: 输入特征 [N, C]
        :param t: 时间步 [N, 1]
        :param labels: 类别标签 [N]，训练时必须传入，推理时可选
        :return: 预测速度 [N, C]
        """
        x = self.shared_proj(x)
        t_emb = self.time_embed(t)
        y = t_emb

        # 训练时：用对应类别的局部流场
        if labels is not None:
            out = []
            for i, label in enumerate(labels):
                feat = x[i:i+1]
                for block in self.class_flow_blocks[f"class_{label.item()}"]:
                    feat = block(feat, y[i:i+1])
                out.append(feat)
            x = torch.cat(out, dim=0)
        # 推理时：默认用全局平均流场（或传入labels指定类别）
        else:
            for blocks in self.class_flow_blocks.values():
                for block in blocks:
                    x = block(x, y)

        velocity = self.final_layer(x, y)
        return velocity

    # ====================== 推理接口 ======================
    def single_step_inference(self, x0):
        """单步推理（和训练分布完全一致）"""
        self.eval()
        with torch.no_grad():
            t = torch.zeros(x0.shape[0], 1, device=x0.device)
            velocity_pred = self.forward(x0, t)
            # 吸引域约束：仅移动到锚点附近，不强行拉到锚点
            x_aligned = x0 + 0.1 * velocity_pred
            return x_aligned

    def get_text_anchors_aligned(self):
        """获取对齐后的文本锚点（推理分类用）"""
        self.eval()
        with torch.no_grad():
            t = torch.zeros(self.text_anchors.shape[0], 1, device=self.text_anchors.device)
            text_aligned = self.text_anchors + 0.1 * self.forward(self.text_anchors, t)
            return text_aligned
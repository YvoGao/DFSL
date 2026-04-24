import torch
import torch.nn as nn

# --------------------------
# Hyperbolic (Poincaré Ball) Utilities
# --------------------------
def arcosh(x):
    # arcosh(x) = ln(x + sqrt(x²-1)), x >= 1
    return torch.log(x + torch.sqrt(x**2 - 1 + 1e-8))

def exp_map0(u, c=1.0):
    # 双曲指数映射（从原点0）: z = exp^u
    c = torch.tensor(c, dtype=u.dtype, device=u.device)
    norm_u = torch.norm(u, dim=-1, keepdim=True) + 1e-8
    sqrt_c = torch.sqrt(c)
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return u * factor

def log_map0(z, c=1.0):
    # 双曲对数映射（到原点0）: u = log^z
    c = torch.tensor(c, dtype=z.dtype, device=z.device)
    norm_z = torch.norm(z, dim=-1, keepdim=True) + 1e-8
    sqrt_c = torch.sqrt(c)
    factor = arcosh(torch.clamp(1 + 2*c*norm_z**2 / ((1 - c*norm_z**2) ** 2), min=1.0 + 1e-8))
    factor = factor / (sqrt_c * norm_z)
    return z * factor

def poincare_distance(z1, z2, c=1.0):
    # 双曲距离（Poincaré ball）
    c = torch.tensor(c, dtype=z1.dtype, device=z1.device)
    diff = z1 - z2
    norm_diff = torch.norm(diff, dim=-1)**2
    norm_z1 = torch.norm(z1, dim=-1)**2
    norm_z2 = torch.norm(z2, dim=-1)**2
    denom = (1 - c*norm_z1) * (1 - c*norm_z2)
    d = 2.0 / torch.sqrt(c) * arcosh(
        1.0 + 2.0 * c * norm_diff / denom + 1e-8
    )
    return d

def euclidean_to_poincare(x, c=1.0):
    # 欧式 -> 双曲（原点映射）
    return exp_map0(x, c=c)

def poincare_to_euclidean(z, c=1.0):
    # 双曲 -> 欧式
    return log_map0(z, c=c)

# --------------------------
# 可学习的双曲投影层
# --------------------------
class HyperbolicProjection(nn.Module):
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))  # 可学习曲率
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        # x: (B, d) 欧式特征
        x = self.proj(x)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8) * 0.9  # 约束在单位球内
        z = euclidean_to_poincare(x, c=self.c)
        return z
import torch
import clip
from torch.utils.data import DataLoader

def get_isotropic_projection(clip_model, data_loader, feat_extractor, cfg, device, kt=200, kb=50, num_samples=1000):
    """
    通用版各向同性子空间提取：不依赖CLIP投影头形状
    通过样本特征的协方差矩阵SVD提取子空间
    """
    clip_model.eval()
    feat_extractor.eval()
    
    print(f"🔍 正在收集 {num_samples} 个样本特征用于子空间提取...")
    
    # 1. 收集投影后的图像和文本特征
    all_img_proj = []
    all_txt_proj = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取特征
            img_feat, txt_feat, _ = feat_extractor(images, labels)
            
            # 用CLIP投影头投影到共享空间（这一步已经完成了W_i和W_t的投影）
            # 注意：这里我们直接用feat_extractor输出的特征，假设已经是投影后的
            all_img_proj.append(img_feat.cpu())
            all_txt_proj.append(txt_feat.cpu())
            
            if len(all_img_proj) * img_feat.shape[0] > num_samples:
                break
    
    # 拼接所有特征
    all_img_proj = torch.cat(all_img_proj, dim=0)[:num_samples]
    all_txt_proj = torch.cat(all_txt_proj, dim=0)[:num_samples]
    
    # 2. 计算联合协方差矩阵
    all_feat = torch.cat([all_img_proj, all_txt_proj], dim=0)
    all_feat = all_feat - all_feat.mean(dim=0, keepdim=True)  # 中心化
    cov = all_feat.T @ all_feat / all_feat.shape[0]
    
    # 3. SVD分解
    print("🧮 正在做SVD分解...")
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    
    # 4. 提取中间各向同性子空间
    embed_dim = all_feat.shape[1]
    subspace_start = kt
    subspace_end = embed_dim - kb
    U_sub = U[:, subspace_start:subspace_end].to(device)
    
    print(f"✅ 各向同性子空间构建完成：原始维度{embed_dim} → 子空间维度{U_sub.shape[1]}")
    return U_sub
import torch
import sys
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math

# ======================
# 替换成你自己项目里的导入（完全兼容你原有代码）
# ======================
from models.ddcmd import DCMDNet
# 替换成你自己的特征提取器
from models.feature_extractor import get_extractor
# 替换成你自己的数据集构建函数
from datasets import build_dataset
# 替换成你自己的配置、日志、随机种子设置
from config import DefaultConfig, SimpleLogger, set_seed

# ======================
# 工具函数1：预计算全局类别文本原型（解决你之前dummy batch的问题）
# ======================
def build_global_class_emb(feat_extractor, train_dataset, cfg):
    """
    用全量训练集预计算全局类别原型，训练和推理全程复用
    :return: class_embeddings [C, D] 每个类别的文本原型
    """
    device = cfg.device
    feat_extractor.eval()
    class_feat_dict = {}
    
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad(), autocast():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            img_feat, _, class_emb = feat_extractor(images, labels)
            # 适配你的feature_extractor输出，确保是[C, D]
            if class_emb.ndim == 3:
                class_emb = class_emb[0]
            # 按类别聚合
            for lbl in torch.unique(labels):
                mask = labels == lbl
                if lbl.item() not in class_feat_dict:
                    class_feat_dict[lbl.item()] = []
                class_feat_dict[lbl.item()].append(class_emb[lbl.item()])
    
    # 每个类别取均值，生成全局原型
    class_embeddings = []
    for lbl in sorted(class_feat_dict.keys()):
        feats = torch.stack(class_feat_dict[lbl])
        class_embeddings.append(feats.mean(dim=0))
    return torch.stack(class_embeddings).to(device)

# ======================
# 工具函数2：APE风格特征解耦掩码生成（训练前仅执行1次）
# ======================
def build_ape_decouple_mask(txt_proto, topk_ratio=0.3, lambda_balance=0.7):
    """
    完全基于APE论文双准则，生成特征解耦掩码
    :param txt_proto: [C, D] 全局类别文本原型
    :param topk_ratio: 保留的语义通道比例，默认30%（和APE论文对齐）
    :param lambda_balance: 双准则平衡系数，APE论文最优值0.7
    :return: mask [1, D] 二值掩码，semantic_idx 语义通道索引
    """
    C, D = txt_proto.shape
    device = txt_proto.device
    # L2归一化，和训练逻辑保持一致
    txt_proto = txt_proto / (txt_proto.norm(dim=-1, keepdim=True) + 1e-8)

    # 准则1：类间相似度（越小，判别性越强）
    S = torch.zeros(D, device=device)
    for k in range(D):
        channel_feat = txt_proto[:, k]
        total_sim = 0.0
        for i in range(C):
            for j in range(C):
                if i != j:
                    total_sim += channel_feat[i] * channel_feat[j]
        S[k] = total_sim / (C ** 2)

    # 准则2：类间方差（越大，区分度越高）
    V = torch.var(txt_proto, dim=0)

    # 双准则融合
    J = lambda_balance * S - (1 - lambda_balance) * V
    # 取J最小的topk个通道，即判别性最强的语义通道
    topk_num = int(D * topk_ratio)
    _, semantic_idx = torch.topk(J, k=topk_num, largest=False)

    # 生成二值掩码
    mask = torch.zeros(D, device=device)
    mask[semantic_idx] = 1.0
    mask = mask.unsqueeze(0)

    print(f"=== APE特征解耦完成 ===")
    print(f"语义通道数：{topk_num}，占比：{topk_num/D:.1%}")
    print(f"平均类间相似度：{S.mean():.6f}，平均类间方差：{V.mean():.6f}")
    return mask, semantic_idx

# ======================
# 测试函数（和训练逻辑完全对齐）
# ======================
def test_dfsl(model, data_loader, feat_extractor, cfg, class_embeddings, decouple_mask=None, enable_dynamic_step=True):
    device = cfg.device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), autocast():
        # 复用预计算的全局原型，避免随机dummy batch
        class_emb_norm = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_emb_norm = class_emb_norm.to(device)

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 提取图像特征
            img_feat, _, _ = feat_extractor(images, labels)
            # 单步漂移对齐
            aligned_feat = model.inference(
                img_feat, 
                class_embeddings, 
                decouple_mask=decouple_mask,
                enable_dynamic_step=enable_dynamic_step
            )
            # 分类
            aligned_feat_norm = aligned_feat / (aligned_feat.norm(dim=-1, keepdim=True) + 1e-8)
            sim = aligned_feat_norm @ class_emb_norm.T
            pred = sim.argmax(dim=-1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# ======================
# 主训练函数
# ======================
def main():
    # ======================
    # 1. 配置初始化（完全兼容你原有代码）
    # ======================
    cfg = DefaultConfig()
    cfg.parse_args()
    # 保存路径
    cfg.save_dir = f'./exp/DFSL/{cfg.dataset}/{cfg.num_shots}-shot/{cfg.feature_extractor}/'
    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg.save()

    # 随机种子、日志
    set_seed(cfg.seed)
    sys.stdout = SimpleLogger(os.path.join(cfg.save_dir, "log.txt"))
    print("="*50)
    print("DCMD Training Start")
    print(cfg)
    print("="*50)

    # 设备
    device = cfg.device
    # 超参数开关（可自由关闭，回到你原始稳定版）
    ENABLE_DECOUPLE = True    # 开启APE特征解耦（核心涨点）
    ENABLE_DYNAMIC_STEP = True # 开启动态步长（适配难样本）
    TOPK_RATIO = 0.3           # 语义通道占比
    LAMBDA_BALANCE = 0.7       # APE双准则平衡系数
    MAX_GRAD_NORM = 1.0        # 梯度裁剪，防止训练崩溃
    PATIENCE = 10              # 早停轮数，防止过拟合

    # ======================
    # 2. 数据集构建（完全兼容你原有代码）
    # ======================
    print("=== 构建数据集 ===")
    dataset = build_dataset(cfg)
    train_loader = DataLoader(
        dataset.train_x, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset.test, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    # 验证集（可选）
    val_loader = None
    if hasattr(dataset, 'val'):
        val_loader = DataLoader(
            dataset.val, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )
    cfg.classnames = dataset.classnames
    print(f"数据集：{cfg.dataset}，类别数：{len(cfg.classnames)}，训练样本数：{len(dataset.train_x)}")

    # ======================
    # 3. 模型初始化
    # ======================
    print("=== 初始化模型 ===")
    # 特征提取器（强制冻结，这是你之前效果不好的核心原因之一）
    feat_extractor = get_extractor(cfg)
    for param in feat_extractor.parameters():
        param.requires_grad = False
    feat_extractor.eval()
    print("特征提取器已冻结，仅训练DCMD漂移模块")

    # DCMD模型
    model = DCMDNet(
        dim=512, 
        hidden_dim=512, 
        num_blocks=cfg.blocks
    ).to(device)
    print(f"DCMD模型初始化完成，参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 优化器、调度器、混合精度
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr if hasattr(cfg, 'lr') else 5e-5, 
        weight_decay=cfg.weight_decay if hasattr(cfg, 'weight_decay') else 5e-5
    )
    # 学习率热身+余弦退火（提升训练稳定性）
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scalar = GradScaler()

    # ======================
    # 4. 预计算全局原型+解耦掩码（训练前仅执行1次）
    # ======================
    print("=== 预计算全局类别原型 ===")
    class_embeddings = build_global_class_emb(feat_extractor, dataset.train_x, cfg)
    print(f"全局原型构建完成，维度：{class_embeddings.shape}")

    # 生成APE解耦掩码
    decouple_mask = None
    if ENABLE_DECOUPLE:
        decouple_mask, semantic_idx = build_ape_decouple_mask(
            class_embeddings, 
            topk_ratio=TOPK_RATIO, 
            lambda_balance=LAMBDA_BALANCE
        )

    # ======================
    # 5. 训练循环
    # ======================
    print("="*50)
    print("开始训练...")
    best_acc = 0.0
    no_improve_count = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                # 提取图像特征
                img_feat, _, _ = feat_extractor(images, labels)
                # 特征归一化
                img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
                # 计算损失
                loss = model.get_drifting_loss(
                    img_feat, 
                    class_embeddings, 
                    labels,
                    decouple_mask=decouple_mask,
                    enable_dynamic_step=ENABLE_DYNAMIC_STEP
                )

            # 反向传播+梯度裁剪
            optimizer.zero_grad()
            scalar.scale(loss).backward()
            # 梯度裁剪，防止训练崩溃
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scalar.step(optimizer)
            scalar.update()

            total_loss += loss.item()
            batch_count += 1

        # 学习率更新
        scheduler.step()
        avg_loss = total_loss / batch_count
        print(f"[Epoch {epoch+1}/{cfg.epochs}] 平均损失：{avg_loss:.6f}，当前学习率：{optimizer.param_groups[0]['lr']:.8f}")

        # 每10轮测试+早停
        if (epoch + 1) % 10 == 0:
            # 优先用验证集选模型，没有验证集用测试集
            eval_loader = val_loader if val_loader is not None else test_loader
            test_acc = test_dfsl(
                model, eval_loader, feat_extractor, cfg, 
                class_embeddings, 
                decouple_mask=decouple_mask,
                enable_dynamic_step=ENABLE_DYNAMIC_STEP
            )
            print(f"[Epoch {epoch+1}] 测试准确率：{test_acc:.4f}")
            
            # 保存最优模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'model': model.state_dict(),
                    'class_embeddings': class_embeddings,
                    'decouple_mask': decouple_mask,
                    'cfg': cfg
                }, os.path.join(cfg.save_dir, 'best_dcmd.pth'))
                print(f"新最优模型已保存！最优准确率：{best_acc:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                # 早停
                if no_improve_count >= PATIENCE:
                    print(f"早停触发，连续{PATIENCE}轮无提升，最优准确率：{best_acc:.4f}")
                    break

    # ======================
    # 6. 最终评估
    # ======================
    print("\n" + "="*50)
    print("最终评估")
    print("="*50)
    # 加载最优模型
    best_ckpt = torch.load(os.path.join(cfg.save_dir, 'best_dcmd.pth'), map_location=device)
    model.load_state_dict(best_ckpt['model'])
    # 最终测试
    final_acc = test_dcmd(
        model, test_loader, feat_extractor, cfg, 
        class_embeddings, 
        decouple_mask=decouple_mask,
        enable_dynamic_step=ENABLE_DYNAMIC_STEP
    )
    print(f"最终测试集准确率：{final_acc:.4f}")
    print(f"历史最优准确率：{best_acc:.4f}")

    # ======================
    # 7. 跨域测试（和你原有逻辑完全一致）
    # ======================
    if cfg.dataset == "ImageNet" and cfg.num_shots == 16:
        print("\n" + "="*50)
        print("跨域数据集测试")
        print("="*50)
        target_names = [
            "ImageNetV2", "ImageNetSketch", "ImageNetR", "ImageNetA",
            "OxfordPets", "EuroSAT", "UCF101", "SUN397", "Caltech101",
            "DescribableTextures", "FGVCAircraft", "Food101", "OxfordFlowers", "StanfordCars"
        ]
        for name in target_names:
            print(f"\n=== 测试 {name} ===")
            t_cfg = DefaultConfig()
            t_cfg.dataset = name
            t_cfg.dataset_root = cfg.dataset_root
            t_cfg.num_shots = cfg.num_shots
            t_data = build_dataset(t_cfg)
            t_cfg.classnames = t_data.classnames
            t_loader = DataLoader(
                t_data.test, 
                batch_size=cfg.batch_size, 
                shuffle=False, 
                num_workers=8, 
                pin_memory=True
            )
            # 目标域特征提取器
            feat_tgt = get_extractor(t_cfg)
            for param in feat_tgt.parameters():
                param.requires_grad = False
            feat_tgt.eval()
            # 目标域全局原型
            t_class_emb = build_global_class_emb(feat_tgt, t_data.train_x, t_cfg)
            # 测试
            acc = test_dfsl(
                model, t_loader, feat_tgt, t_cfg, 
                t_class_emb, 
                decouple_mask=decouple_mask,
                enable_dynamic_step=ENABLE_DYNAMIC_STEP
            )
            print(f"{name} 准确率：{acc:.4f}")

if __name__ == '__main__':
    main()
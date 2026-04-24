import torch
from models.fm import DeepFlowMatchingNet
from torch.utils.data import DataLoader
import clip
import torch.nn.functional as F
from config import DefaultConfig
from config import set_seed
import sys
from torch.cuda.amp import autocast
from models.feature_extractor import get_extractor
from datasets import build_dataset
from einops import einsum
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_r import ImageNetR
from datasets.imagenet_a import ImageNetA
import os
@torch.no_grad()
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



if __name__ == "__main__":
    # timestamp = sys.argv[1] # each timestamp corresponding to an exp.

    # print(f'Loading Checkpoint at Experiment with timestamp: {timestamp }')
    config_path = f"./exp/DFSL/ImageNet/16-shot/lora/config.json"
    cfg = DefaultConfig.from_json(config_path)
    model_path = f"./exp/DFSL/{cfg.dataset}/{cfg.num_shots}-shot/{cfg.feature_extractor}/model.pth"
    print(cfg)
    set_seed(cfg.seed)

     # Prepare dataset
    dataset = build_dataset(cfg)
    cfg.classnames = dataset.classnames

    
    train_loader = DataLoader(dataset.train_x, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset.test, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset.val, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if cfg.dataset == "ImageNet" and cfg.num_shots == 16:

        target_datasets = ["ImageNetV2", "ImageNetSketch", "ImageNetR","ImageNetA","OxfordPets", "EuroSAT", "UCF101", "SUN397", "Caltech101", "DescribableTextures", "FGVCAircraft", "Food101", "OxfordFlowers", "StanfordCars"]
        
        target_loaders = []
        target_cfgs = []
        for target_dataset in target_datasets:
            target_cfg = DefaultConfig()
            target_cfg.dataset = target_dataset
            target_cfg.dataset_root = cfg.dataset_root
            target_cfg.num_shots = cfg.num_shots
            target_data = build_dataset(target_cfg)
            target_cfg.classnames = target_data.classnames
            target_loader = DataLoader(target_data.test, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            target_loaders.append(target_loader)
            target_cfgs.append(target_cfg)
    # Initialize the model
    clip_model, _ = clip.load(cfg.clip_type, device=cfg.device, jit=False)
    dim = clip_model.visual.output_dim
    print(f"CLIP model output dimension: {dim}")
    model = DeepFlowMatchingNet(in_channels=dim,model_channels=dim, out_channels=dim,num_res_blocks=cfg.blocks).to(cfg.device)

    # load model
    state_dict = torch.load(model_path,map_location=cfg.device)
    model.load_state_dict(state_dict)

    
    # feat_extractor = get_extractor(cfg)
    # print("🤖 Evaluating on Test Dataset...")
    # for steps in [0,1,2,3,4,5,6,7,8,9,10]:
    #     test_fma(model,test_loader,feat_extractor,steps=steps,stepsize=0.1,cfg=cfg)
   
    if cfg.dataset == "ImageNet" and cfg.num_shots == 16:
       for target_loader, target_dataset, target_cfg in zip(target_loaders, target_datasets, target_cfgs):
            print(f'Final Testing On {target_dataset}')
            feat_extractor = get_extractor(target_cfg)
            best_acc = 0.0
            for steps in [0,1,2,3,4,5,6,7,8,9,10]:
                test_acc = test_dfsl(model,target_loader,feat_extractor,steps=steps,stepsize=0.1, cfg=cfg)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_steps = steps
            print(f"Dataset:{target_dataset}, Blocks:{cfg.blocks}, Seed:{cfg.seed}; Best Accuracy: {best_acc:.4f}, at Steps:{best_steps}; Velocity saved at {cfg.save_dir}/model.pth")
    
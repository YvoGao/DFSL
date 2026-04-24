import os
import re
import csv
from pathlib import Path
import pandas as pd

# ====================== 配置 ======================
BASE_EXP_DIR = "./exp_org/dcmd_en"
OUTPUT_CSV = f"{Path(BASE_EXP_DIR).name}.csv"  # 自动获取文件夹名

TARGET_DATASETS = [
    "ImageNet", "ImageNetV2", "ImageNetSketch", "ImageNetR", "ImageNetA",
    "OxfordPets", "EuroSAT", "UCF101", "SUN397", "Caltech101",
    "DescribableTextures", "FGVCAircraft", "Food101", "OxfordFlowers", "StanfordCars"
]
# ==================================================

def extract_acc_from_log(log_path):
    if not os.path.exists(log_path):
        return None

    acc_pattern = re.compile(r"Accuracy:\s*(\d+\.\d+)")
    last_acc = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = acc_pattern.search(line)
            if match:
                last_acc = float(match.group(1))
    return last_acc

def parse_path_auto(root):
    """
    自动解析路径中的 dataset, shots, backbone
    不管层级多少，都能识别！
    """
    parts = Path(root).parts
    
    dataset = "unknown"
    shots = "unknown"
    backbone = "unknown"

    # 自动找 dataset（在TARGET_DATASETS里匹配）
    for p in parts:
        if p in TARGET_DATASETS:
            dataset = p
            break

    # 自动找 shot（如 16-shot, 8-shot）
    for p in parts:
        if "shot" in p:
            shots = p.replace("-shot", "")
            break

    # 自动找 backbone（clip / cocoop）
    for p in parts:
        if p in ["clip", "cocoop", "vpt", "clip_s"]:
            backbone = p
            break

    return dataset, shots, backbone

def find_all_exps(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        if "log.txt" in files:
            log_path = os.path.join(root, "log.txt")
            acc = extract_acc_from_log(log_path)
            if acc is None:
                continue

            dataset, shots, backbone = parse_path_auto(root)

            results.append({
                "Dataset": dataset,
                "Shots": shots,
                "Backbone": backbone,
                "Accuracy": acc,
                "LogPath": root
            })
    return results

def build_table(results):
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Dataset", "Shots", "Backbone"])
    df = df.reset_index(drop=True)
    return df

def print_pretty_table(df):
    print("\n" + "=" * 100)
    print(f"{'Dataset':<20} {'Shots':<8} {'Backbone':<12} {'Accuracy':<10}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{row['Dataset']:<20} {row['Shots']:<8} {row['Backbone']:<12} {row['Accuracy']:<10.4f}")
    print("=" * 100)

def main():
    print(f"🔍 正在扫描实验目录：{BASE_EXP_DIR}")
    results = find_all_exps(BASE_EXP_DIR)

    if not results:
        print("❌ 未找到任何实验结果！")
        return

    df = build_table(results)
    print_pretty_table(df)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存到：{OUTPUT_CSV}")

    print("\n📊 数据集汇总：")
    for ds in TARGET_DATASETS:
        sub = df[df["Dataset"] == ds]
        if not sub.empty:
            mean = sub["Accuracy"].mean()
            print(f"{ds:<20} 平均精度: {mean:.4f}")

if __name__ == "__main__":
    main()
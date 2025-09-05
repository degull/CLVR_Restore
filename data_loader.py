# data_loader.py
import os, random, csv
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
import pandas as pd
import torch

# ✅ 기본 transform
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def sample_paths(paths, n=160):
    if len(paths) > n:
        return random.sample(paths, n)
    return paths

# -------------------------
# PairedDataset (Rain, Snow, Blur)
# -------------------------
class PairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, type_label, strength_label=1,
                 ext=(".png", ".jpg", ".bmp", ".tif", ".jpeg"),
                 limit=160, transform=None):
        self.input_paths = sorted([p for p in glob(os.path.join(input_dir, "*")) if p.lower().endswith(ext)])
        self.target_paths = sorted([p for p in glob(os.path.join(target_dir, "*")) if p.lower().endswith(ext)])

        print(f"📂 Loading dataset: {input_dir} vs {target_dir}")
        print(f"   found {len(self.input_paths)} input, {len(self.target_paths)} target")

        self.input_paths = sample_paths(self.input_paths, limit)
        self.target_paths = sample_paths(self.target_paths, limit)

        self.transform = transform if transform is not None else default_transform
        self.type_label = type_label
        self.strength_label = strength_label

    def __len__(self):
        return min(len(self.input_paths), len(self.target_paths))

    def __getitem__(self, idx):
        x = Image.open(self.input_paths[idx]).convert("RGB")
        y = Image.open(self.target_paths[idx]).convert("RGB")
        x, y = self.transform(x), self.transform(y)
        return x, y, self.type_label, self.strength_label


# -------------------------
# CSV 기반 Dataset (SIDD 전용)
# -------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, type_label, limit=160, transform=None):
        self.df = pd.read_csv(csv_file)
        if limit > 0 and len(self.df) > limit:
            self.df = self.df.sample(limit, random_state=42)

        self.transform = transform if transform is not None else default_transform
        self.type_label = type_label

        print(f"📂 Loading SIDD CSV: {csv_file}")
        print(f"   found {len(self.df)} pairs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        noisy = Image.open(row["dist_img"]).convert("RGB")
        clean = Image.open(row["ref_img"]).convert("RGB")

        # strength 라벨 자동 추출
        path = row["dist_img"]
        if "_L" in path:
            strength_label = 0
        elif "_N" in path:
            strength_label = 1
        elif "_H" in path:
            strength_label = 2
        else:
            strength_label = 1  # 기본값 medium

        return self.transform(noisy), self.transform(clean), self.type_label, strength_label


# -------------------------
# 통합 로더
# -------------------------
def get_train_datasets(base_dir="E:/CLVR_Restore/data", limit=160, with_labels=False):
    datasets = []
    paths = [
        ("Rain100H", "rain100H/train/rain", "rain100H/train/norain", 0, 2),
        ("Rain100L", "rain100L/train/rain", "rain100L/train/norain", 0, 0),
        ("HIDE", "HIDE/train", "HIDE/GT", 2, 1),        # Blur
        ("CSD", "CSD/Train/Snow", "CSD/Train/Gt", 1, 1) # Snow
    ]

    for name, inp, tgt, t_label, s_label in paths:
        input_dir = os.path.join(base_dir, inp)
        target_dir = os.path.join(base_dir, tgt)

        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"❌ {name} dataset missing: {input_dir} / {target_dir}")
            continue

        ds = PairedDataset(
            input_dir, target_dir,
            type_label=t_label,
            strength_label=s_label,
            limit=limit
        )
        # with_labels=True면 (x, y, type, strength)로 변환
        if with_labels:
            ds = [(x, y, torch.tensor(t_label), torch.tensor(s_label)) for x, y, _, _ in ds]

        print(f"✅ {name} dataset loaded: {len(ds)} samples")
        datasets.append(ds)

    # SIDD CSV
    sidd_csv = os.path.join(base_dir, "SIDD", "sidd_pairs.csv")
    if os.path.exists(sidd_csv):
        ds = CSVDataset(sidd_csv, type_label=3, limit=limit)
        if with_labels:
            ds = [(x, y, torch.tensor(3), torch.tensor(strength)) for x, y, _, strength in ds]
        print(f"✅ SIDD dataset loaded: {len(ds)} samples")
        datasets.append(ds)
    else:
        print(f"❌ SIDD CSV not found: {sidd_csv}")

    return datasets

import os, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image

# ✅ 모든 이미지를 256×256으로 Resize + ToTensor
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),   # 크기 통일
    transforms.ToTensor()
])

def sample_paths(paths, n=160):
    if len(paths) > n:
        return random.sample(paths, n)
    return paths

class PairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, ext=(".png", ".jpg", ".bmp"), limit=160, transform=None):
        self.input_paths = sorted([p for p in glob(os.path.join(input_dir, "*")) if p.lower().endswith(ext)])
        self.target_paths = sorted([p for p in glob(os.path.join(target_dir, "*")) if p.lower().endswith(ext)])

        # dataset 크기 맞추기 (랜덤 샘플링)
        self.input_paths = sample_paths(self.input_paths, limit)
        self.target_paths = sample_paths(self.target_paths, limit)

        # transform 지정 (없으면 default)
        self.transform = transform if transform is not None else default_transform

    def __len__(self):
        return min(len(self.input_paths), len(self.target_paths))

    def __getitem__(self, idx):
        x = Image.open(self.input_paths[idx]).convert("RGB")
        y = Image.open(self.target_paths[idx]).convert("RGB")

        x = self.transform(x)
        y = self.transform(y)

        return x, y


def get_train_datasets(base_dir="E:/CLVR_Restore/data", limit=160):
    datasets = []

    # Rain100H
    datasets.append(PairedDataset(
        os.path.join(base_dir, "rain100H/train/rain"),
        os.path.join(base_dir, "rain100H/train/norain"),
        limit=limit
    ))

    # Rain100L
    datasets.append(PairedDataset(
        os.path.join(base_dir, "rain100L/train/rain"),
        os.path.join(base_dir, "rain100L/train/norain"),
        limit=limit
    ))

    # HIDE
    datasets.append(PairedDataset(
        os.path.join(base_dir, "HIDE/train"),
        os.path.join(base_dir, "HIDE/GT"),
        limit=limit
    ))

    # CSD
    datasets.append(PairedDataset(
        os.path.join(base_dir, "CSD/Train/Snow"),
        os.path.join(base_dir, "CSD/Train/Gt"),
        limit=limit
    ))

    # SIDD ⚠️ noisy/GT 페어링 수정 필요
    # → 현재는 예시로 하나의 폴더만 지정했는데,
    # 실제로는 CSV (noisy,gt 경로) 읽어서 PairDataset을 구성하는 게 더 안정적임
    # 지금 구조에서는 일단 첫 시퀀스만 사용
    datasets.append(PairedDataset(
        os.path.join(base_dir, "SIDD/Data/0002_001_S6_00100_00020_3200_N"),
        os.path.join(base_dir, "SIDD/Data/0002_001_S6_00100_00020_3200_N"),
        limit=limit
    ))

    return datasets

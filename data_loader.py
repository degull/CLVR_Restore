import os, random
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image

# ✅ 기본 transform
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def sample_paths(paths, n=160):
    if len(paths) > n:
        return random.sample(paths, n)
    return paths

class PairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, ext=(".png", ".jpg", ".bmp", ".tif", ".jpeg"),
                 limit=160, transform=None):
        self.input_paths = sorted([p for p in glob(os.path.join(input_dir, "*")) if p.lower().endswith(ext)])
        self.target_paths = sorted([p for p in glob(os.path.join(target_dir, "*")) if p.lower().endswith(ext)])

        print(f"📂 Loading dataset: {input_dir} vs {target_dir}")
        print(f"   found {len(self.input_paths)} input, {len(self.target_paths)} target")

        self.input_paths = sample_paths(self.input_paths, limit)
        self.target_paths = sample_paths(self.target_paths, limit)
        self.transform = transform if transform is not None else default_transform

    def __len__(self):
        return min(len(self.input_paths), len(self.target_paths))

    def __getitem__(self, idx):
        x = Image.open(self.input_paths[idx]).convert("RGB")
        y = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(x), self.transform(y)


def get_train_datasets(base_dir="E:/CLVR_Restore/data", limit=160):
    datasets = []
    paths = [
        ("Rain100H", "rain100H/train/rain", "rain100H/train/norain"),
        ("Rain100L", "rain100L/train/rain", "rain100L/train/norain"),
        ("HIDE", "HIDE/train", "HIDE/GT"),
        ("CSD", "CSD/Train/Snow", "CSD/Train/Gt"),
        #("SIDD", "SIDD/Data", "SIDD/Data"),
        ("GoPro", "GOPRO/train/blur", "GOPRO/train/sharp"),
        #("BSDS500", "BSDS500/images/train", "BSDS500/ground_truth/train"),
        #("Classic5", "classic5/gray/qf_10", "classic5/refimgs"),
        #("LIVE1", "live1/qf_10", "live1/refimgs"),
    ]

    for name, inp, tgt in paths:
        input_dir = os.path.join(base_dir, inp)
        target_dir = os.path.join(base_dir, tgt)

        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"❌ {name} dataset missing: {input_dir} / {target_dir}")
            continue

        ds = PairedDataset(input_dir, target_dir, limit=limit)
        print(f"✅ {name} dataset loaded: {len(ds)} samples")
        datasets.append(ds)

    return datasets

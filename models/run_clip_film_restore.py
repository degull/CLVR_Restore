# file: run_clip_film_restore.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torchvision import transforms

from models.restormer_volterra_film import RestormerVolterraFiLM


# ---------------- CLIP Adapter ---------------- #
class CLIPAdapter(nn.Module):
    """ CLIP Vision Encoder → condition vector z """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", cond_dim=128, device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP (학습 안 함)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # projection: hidden_size → cond_dim
        hidden_dim = self.clip_model.config.vision_config.hidden_size
        self.proj = nn.Linear(hidden_dim, cond_dim).to(device)

    def forward(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output  # [B, hidden_dim]
        z = self.proj(pooled)  # [B, cond_dim]
        return z


# ---------------- End-to-End Restore Pipeline ---------------- #
class CLVRRestore(nn.Module):
    def __init__(self, cond_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.clip_adapter = CLIPAdapter(cond_dim=cond_dim, device=device)
        self.restorer = RestormerVolterraFiLM(cond_dim=cond_dim).to(device)

        # ✅ 입력 정규화 (학습 시와 일치)
        self.input_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def forward(self, image: Image.Image):
        """ image: PIL.Image, returns restored tensor """
        # 1. condition vector z from CLIP
        z = self.clip_adapter(image)
        print(f"🔹 Condition vector — mean: {z.mean():.4f}, std: {z.std():.4f}")

        # 2. 원본 해상도로 tensor 변환 + 정규화
        x = self.input_tfm(image).unsqueeze(0).to(self.device)  # [1,3,H,W]

        # 3. restoration
        with torch.no_grad():
            out = self.restorer(x, z)

        # 4. 역정규화 (복원된 결과를 0~1 범위로 되돌림)
        out = (out * 0.5) + 0.5
        out = out.clamp(0, 1)
        return out


# ---------------- Example Run ---------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = CLVRRestore(cond_dim=128, device=device)

    # ✅ 학습된 checkpoint 로드
    ckpt_path = "E:/CLVR_Restore/checkpoints/stage3_epoch10_modefixed_stratgeneralization_ssim0.8892_psnr34.02.pth"
    ckpt = torch.load(ckpt_path, map_location=device)

    # ✅ 복합 ckpt 처리
    if "restorer" in ckpt:
        pipeline.restorer.load_state_dict(ckpt["restorer"])
        print("✅ Loaded model weights from ckpt['restorer']")
    elif "model" in ckpt:
        pipeline.restorer.load_state_dict(ckpt["model"])
        print("✅ Loaded model weights from ckpt['model']")
    else:
        pipeline.restorer.load_state_dict(ckpt)
        print("✅ Loaded model weights directly from state_dict")

    print(f"📂 Checkpoint loaded from: {ckpt_path}")

    # ✅ 입력 이미지 로드
    img_path = "E:/CLVR_Restore/data/CSD/Test/Snow/4.tif"
    img = Image.open(img_path).convert("RGB")
    print(f"🖼️ Input image loaded from: {img_path}")

    # ✅ 복원 수행
    pipeline.eval()
    restored = pipeline(img)  # [1,3,H,W]
    print("✅ Restoration forward pass complete")
    print(f"Restored tensor shape: {restored.shape}")

    # ✅ 출력 후처리 및 저장
    out = restored.squeeze(0).cpu()
    out_img = transforms.ToPILImage()(out)

    save_dir = "E:/CLVR_Restore/check"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "restored_with_checkpoint_fixed.png")
    out_img.save(save_path)

    # ✅ 통계 출력
    mean_val = float(out.mean())
    min_val = float(out.min())
    max_val = float(out.max())

    print(f"✅ Final restored image saved at: {save_path}")
    print(f"📊 Output statistics — mean: {mean_val:.4f}, min: {min_val:.4f}, max: {max_val:.4f}")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from models.run_clip_film_restore import CLIPAdapter

def test_clip_adapter():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ CLIPAdapter 로드
    clip_adapter = CLIPAdapter(
        clip_model_name="openai/clip-vit-base-patch32",
        cond_dim=128,
        device=device
    )

    # 2️⃣ 테스트용 이미지 (아무 RGB 이미지 OK)
    img_path = "E:/CLVR_Restore/data/CSD/Test/Snow/4.tif"
    image = Image.open(img_path).convert("RGB")

    # 3️⃣ Forward
    with torch.no_grad():
        z = clip_adapter(image)

    # 4️⃣ 출력 확인
    print("✅ CLIP Adapter forward 성공")
    print(f"Condition vector shape: {tuple(z.shape)}")
    print(f"평균: {z.mean().item():.4f}, 표준편차: {z.std().item():.4f}")

if __name__ == "__main__":
    test_clip_adapter()

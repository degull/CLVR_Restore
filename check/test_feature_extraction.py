import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
from models.restormer_volterra_film import RestormerVolterraFiLM

def test_feature_extraction():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 더미 입력 생성
    img = torch.randn(1, 3, 256, 256).to(device)
    cond = torch.randn(1, 128).to(device)   # FiLM용 condition vector

    # 2. 모델 로드
    model = RestormerVolterraFiLM(cond_dim=128).to(device)
    model.eval()

    # 3. Forward 테스트
    with torch.no_grad():
        out = model(img, cond)

    # 4. 출력 확인
    print("✅ Forward pass 성공")
    print(f"입력 shape: {tuple(img.shape)}")
    print(f"출력 shape: {tuple(out.shape)}")
    print(f"출력 평균값: {out.mean().item():.4f}, 최소: {out.min().item():.4f}, 최대: {out.max().item():.4f}")

    # 5. Gradient 확인
    model.train()
    img.requires_grad = True
    out = model(img, cond)
    loss = out.mean()
    loss.backward()
    print("✅ Gradient 정상 계산됨 (첫 conv grad 평균):", model.patch_embed.weight.grad.mean().item())

if __name__ == "__main__":
    test_feature_extraction()

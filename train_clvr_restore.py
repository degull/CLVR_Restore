# E:\CLVR_Restore\train_clvr_restore.py
""" import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric

from transformers import CLIPModel, CLIPProcessor, AutoModel

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition
from data_loader import get_train_datasets


# ---------------------------
# Metrics
# ---------------------------
def compute_ssim(pred, gt):
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt   = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals = []
    for i in range(pred.shape[0]):
        vals.append(
            ssim_metric(gt[i], pred[i], channel_axis=2, data_range=1.0)
        )
    return torch.tensor(sum(vals) / len(vals))

def compute_psnr(pred, gt):
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt   = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals = []
    for i in range(pred.shape[0]):
        vals.append(
            psnr_metric(gt[i], pred[i], data_range=1.0)
        )
    return torch.tensor(sum(vals) / len(vals))


# ---------------------------
# Main Training
# ---------------------------
def main(stage=1, epochs=2, batch_size=2, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16"
    ).vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size  # 768

    # 2. Mistral-7B-Instruct (fp16)
    llm_model = AutoModel.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",

    )
    llm_dim = llm_model.config.hidden_size  # 4096

    if stage == 1:
        # Stage1 → backbone 학습, LLM freeze
        for p in llm_model.parameters():
            p.requires_grad = False

    # 3. Bridge
    vision_to_llm = VisionToLLMAdapter(
        clip_dim=clip_dim, llm_dim=llm_dim
    ).to(device)
    llm_to_cond   = LLMToCondition(
        llm_dim=llm_dim, cond_dim=128
    ).to(device)

    # 4. Restorer Backbone
    restorer = RestormerVolterraFiLM(cond_dim=128).to(device)

    if stage == 1:
        params = list(restorer.parameters())
    else:
        # Stage2 → backbone freeze, bridge 학습
        for p in restorer.parameters():
            p.requires_grad = False
        params = list(vision_to_llm.parameters()) + list(llm_to_cond.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    # 5. Dataset
    datasets = get_train_datasets(limit=160)  # 각 데이터셋 160개 샘플로 균형
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(epochs):
        for img, gt in dataloader:
            img, gt = img.to(device), gt.to(device)

            # (1) CLIP Vision Embedding
            pil_imgs = [TF.to_pil_image(i.cpu()) for i in img]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                e_clip = clip_model(**inputs).pooler_output  # [B, 768]

            # (2) Vision->LLM
            vision_embed = vision_to_llm(e_clip)  # [B, 4096]

            # (3) LLM Forward
            llm_out = llm_model(inputs_embeds=vision_embed.unsqueeze(1))
            h_llm = llm_out.last_hidden_state[:, -1, :]  # [B, 4096]

            # (4) LLM->Condition
            z = llm_to_cond(h_llm)  # [B, 128]

            # (5) Restore
            pred = restorer(img, z)

            # (6) Loss
            ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
            psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
            loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Stage {stage}] Epoch {epoch}, Loss {loss.item():.4f}, "
              f"PSNR {psnr_val:.2f}, SSIM {ssim_val:.4f}")


if __name__ == "__main__":
    # Stage1: Backbone 학습
    main(stage=1, epochs=10)

    # Stage2: Bridge 학습
    main(stage=2, epochs=10)
 """

# pth 저장코드

# E:/CLVR_Restore/train_clvr_restore.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm   # ✅ 진행바 추가

from transformers import CLIPModel, CLIPProcessor, AutoModel
from torch.amp import autocast, GradScaler  # ✅ 최신 AMP API

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition
from data_loader import get_train_datasets


# ---------------------------
# Metrics
# ---------------------------
def compute_ssim(pred, gt):
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt   = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals = [
        ssim_metric(gt[i], pred[i], channel_axis=2, data_range=1.0)
        for i in range(pred.shape[0])
    ]
    return torch.tensor(sum(vals) / len(vals))

def compute_psnr(pred, gt):
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt   = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals = [
        psnr_metric(gt[i], pred[i], data_range=1.0)
        for i in range(pred.shape[0])
    ]
    return torch.tensor(sum(vals) / len(vals))


# ---------------------------
# Main Training
# ---------------------------
def main(stage=1, epochs=100, batch_size=1, lr=1e-4, save_dir="E:/CLVR_Restore/checkpoints"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size  # 768

    # 2. LLM (Mistral 7B) with gradient checkpointing
    llm_model = AutoModel.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm_model.gradient_checkpointing_enable()  # ✅ gradient checkpointing
    llm_dim = llm_model.config.hidden_size

    if stage == 1:
        for p in llm_model.parameters():
            p.requires_grad = False

    # 3. Bridge
    vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(device)
    llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(device)

    # 4. Restorer Backbone
    restorer = RestormerVolterraFiLM(cond_dim=128).to(device)

    if stage == 1:
        params = list(restorer.parameters())
    else:
        for p in restorer.parameters():
            p.requires_grad = False
        params = list(vision_to_llm.parameters()) + list(llm_to_cond.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    scaler = GradScaler("cuda")  # ✅ AMP scaler (최신 API)

    # 5. Dataset
    datasets = get_train_datasets(limit=160)
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(1, epochs+1):
        pbar = tqdm(dataloader, desc=f"[Stage {stage}] Epoch {epoch}/{epochs}", leave=True)
        for step, (img, gt) in enumerate(pbar, start=1):
            img, gt = img.to(device), gt.to(device)

            # (1) CLIP Vision Embedding
            pil_imgs = [T.ToPILImage()(i.cpu()) for i in img]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                e_clip = clip_model(**inputs).pooler_output  # [B, 768]

            # (2) Vision->LLM
            vision_embed = vision_to_llm(e_clip).to(dtype=torch.float16)  # ✅ match fp16

            with autocast("cuda"):  # ✅ AMP autocast
                # (3) LLM Forward
                llm_out = llm_model(inputs_embeds=vision_embed.unsqueeze(1))
                h_llm = llm_out.last_hidden_state[:, -1, :]

                # (4) LLM->Condition
                z = llm_to_cond(h_llm)

                # (5) Restore
                pred = restorer(img, z)

                # (6) Loss
                ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
                psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
                loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

            optimizer.zero_grad()
            scaler.scale(loss).backward()   # ✅ AMP backward
            scaler.step(optimizer)
            scaler.update()

            # 진행바에 현재 상태 표시
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val:.4f}"
            })

        # ✅ Epoch 종료 로그
        print(f"\n[Stage {stage}] Epoch {epoch} finished → Loss {loss.item():.4f}, "
              f"PSNR {psnr_val:.2f}, SSIM {ssim_val:.4f}")

        # ✅ 체크포인트 저장
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}_ssim{ssim_val:.4f}_psnr{psnr_val:.2f}.pth")
        torch.save({
            "epoch": epoch,
            "restorer": restorer.state_dict(),
            "vision_to_llm": vision_to_llm.state_dict(),
            "llm_to_cond": llm_to_cond.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)
        print(f"✓ Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    # Stage1: Backbone 학습
    main(stage=1, epochs=100, batch_size=1)

    # Stage2: Bridge 학습
    main(stage=2, epochs=100, batch_size=1)

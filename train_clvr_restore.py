# E:/CLVR_Restore/train_clvr_restore.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM
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
    vals = [ssim_metric(gt[i], pred[i], channel_axis=2, data_range=1.0)
            for i in range(pred.shape[0])]
    return torch.tensor(sum(vals) / len(vals))

def compute_psnr(pred, gt):
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt   = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals = [psnr_metric(gt[i], pred[i], data_range=1.0)
            for i in range(pred.shape[0])]
    return torch.tensor(sum(vals) / len(vals))


# ---------------------------
# Main Training
# ---------------------------
def main(stage=1, epochs=100, batch_size=1, lr=1e-4,
         save_dir="E:/CLVR_Restore/checkpoints", lambda_cons=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size  # 768

    # 2. LLM (Mistral 7B, causal LM for logits)
    llm_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm_model.gradient_checkpointing_enable()
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
    scaler = GradScaler("cuda")

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

            # -------- augmentation 2 views --------
            aug = T.RandomApply([
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.RandomRotation(5)
            ], p=0.5)

            pil_imgs1 = [T.ToPILImage()(i.cpu()) for i in img]
            pil_imgs2 = [aug(T.ToPILImage()(i.cpu())) for i in img]

            inputs1 = processor(images=pil_imgs1, return_tensors="pt").to(device)
            inputs2 = processor(images=pil_imgs2, return_tensors="pt").to(device)

            with torch.no_grad():
                e_clip1 = clip_model(**inputs1).pooler_output
                e_clip2 = clip_model(**inputs2).pooler_output

            vision_embed1 = vision_to_llm(e_clip1).to(dtype=torch.float16)
            vision_embed2 = vision_to_llm(e_clip2).to(dtype=torch.float16)

            with autocast("cuda"):
                # --- LLM forward ---
                out1 = llm_model(inputs_embeds=vision_embed1.unsqueeze(1))
                out2 = llm_model(inputs_embeds=vision_embed2.unsqueeze(1))

                logits1 = out1.logits  # [B, T, V]
                logits2 = out2.logits

                # Consistency loss (KL)
                p = torch.log_softmax(logits1, dim=-1)
                q = torch.softmax(logits2, dim=-1)
                cons_loss = F.kl_div(p, q, reduction="batchmean")

                # --- Use 1st view for restoration ---
                h_llm = out1.hidden_states[-1][:, -1, :] if out1.hidden_states else out1.last_hidden_state[:, -1, :]
                z = llm_to_cond(h_llm.to(dtype=torch.float32))
                pred = restorer(img, z)

                # Restoration loss
                ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
                psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
                rest_loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

                # Total loss
                loss = rest_loss + lambda_cons * cons_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rest": f"{rest_loss.item():.4f}",
                "cons": f"{cons_loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val:.4f}"
            })

        # ---- Epoch end ----
        print(f"\n[Stage {stage}] Epoch {epoch} → Loss {loss.item():.4f}, Rest {rest_loss.item():.4f}, Cons {cons_loss.item():.4f}, "
              f"PSNR {psnr_val:.2f}, SSIM {ssim_val:.4f}")

        # Save ckpt
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
    main(stage=1, epochs=100, batch_size=1, lambda_cons=0.1)

    # Stage2: Bridge 학습
    main(stage=2, epochs=100, batch_size=1, lambda_cons=0.1)

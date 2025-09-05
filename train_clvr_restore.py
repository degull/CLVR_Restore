import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from torch.amp import autocast, GradScaler

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition
from data_loader import get_train_datasets
from transformers import AutoModel, AutoTokenizer

# ---------------- Metrics ---------------- #
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


# ---------------- Main Training ---------------- #
def main(stage=1, epochs=100, batch_size=1, lr=1e-4,
         save_dir="E:/CLVR_Restore/checkpoints", lambda_cons=0.1, log_interval=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size  # 768

    # 2. LLM (small debug model: OPT-125M, or replace with mistralai/Mistral-7B-Instruct-v0.2)
    llm_name = "facebook/opt-125m"  # ⚠️ 바꿀 수 있음
    llm_model = AutoModel.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False)
    llm_model.gradient_checkpointing_enable()
    llm_dim = llm_model.config.hidden_size

    if stage == 1:
        for p in llm_model.parameters():
            p.requires_grad = False

    # 3. Bridge
    vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(device)
    llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(device)

    # 4. Restorer
    restorer = RestormerVolterraFiLM(cond_dim=128).to(device)

    if stage == 1:
        params = list(restorer.parameters())
    elif stage == 2:
        for p in restorer.parameters(): p.requires_grad = False
        params = list(vision_to_llm.parameters()) + list(llm_to_cond.parameters())
    else:  # Stage 3 joint finetuning
        for p in restorer.parameters(): p.requires_grad = True
        params = list(restorer.parameters()) + list(vision_to_llm.parameters()) + list(llm_to_cond.parameters())
        lr = lr * 0.1  # smaller LR for joint finetuning

    optimizer = torch.optim.Adam(params, lr=lr)
    scaler = GradScaler("cuda")

    # 5. Dataset
    datasets = get_train_datasets(limit=160)
    print(f"✅ Loaded datasets: {len(datasets)}")
    for i, d in enumerate(datasets):
        print(f"   ↳ Dataset {i} size: {len(d)}")

    if len(datasets) == 0:
        print("❌ ERROR: No datasets loaded. Check your data paths!")
        return

    train_dataset = ConcatDataset(datasets)
    print(f"✅ Total training samples: {len(train_dataset)}")

    if len(train_dataset) == 0:
        print("❌ ERROR: Training dataset is empty.")
        return

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ Dataloader batches: {len(dataloader)}")


    # ---------------- Training Loop ---------------- #
    for epoch in range(1, epochs+1):
        print(f"\n🚀 Starting Stage {stage} Epoch {epoch}/{epochs}")
        pbar = tqdm(dataloader, desc=f"[Stage {stage}] Epoch {epoch}/{epochs}", leave=True)

        for step, (img, gt) in enumerate(pbar, start=1):
            img, gt = img.to(device), gt.to(device)

            pil_imgs = [T.ToPILImage()(i.cpu()) for i in img]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                e_clip = clip_model(**inputs).pooler_output

            vision_embed = vision_to_llm(e_clip).to(dtype=torch.float16)

            # -------- LLM 입력: text prefix + vision embed --------
            prefix = "Distortion type reasoning:"
            prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
            prefix_emb = llm_model.get_input_embeddings()(prefix_ids)  # [1, T, d]
            prefix_emb = prefix_emb.expand(img.size(0), -1, -1).contiguous()

            combined = torch.cat([prefix_emb, vision_embed.unsqueeze(1)], dim=1)  # [B, T+1, d]

            with autocast("cuda"):
                out = llm_model(inputs_embeds=combined)

                # ✅ AutoModel always returns hidden states
                h_llm = out.last_hidden_state[:, -1, :]  

                z = llm_to_cond(h_llm.to(dtype=torch.float32))
                pred = restorer(img, z)


                ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
                psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
                rest_loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

                loss = rest_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val:.4f}"
            })

        # ---- Save ckpt ----
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
    main(stage=1, epochs=50, batch_size=1)

    # Stage2: Bridge 학습
    main(stage=2, epochs=50, batch_size=1)

    # Stage3: Joint finetune
    main(stage=3, epochs=20, batch_size=1)

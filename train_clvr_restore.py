# train_clvr_restore.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition
from models.aux_classifier import AuxClassifier
from models.llm_lora import apply_lora_to_llm
from data_loader import get_train_datasets


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
def main(stage=1, epochs=5, batch_size=1, lr=1e-4,
         save_dir="E:/CLVR_Restore/checkpoints", lambda_aux=0.2, log_interval=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        use_safetensors=True  # ✅ safetensors 강제
    ).vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size

    # 2. LLM (OPT small for debug; replace with LLaMA/Mistral for real exp)
    llm_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
    llm_model.gradient_checkpointing_enable()
    llm_dim = llm_model.config.hidden_size

    if stage in [2, 3]:
        llm_model = apply_lora_to_llm(llm_model)  # ✅ LoRA 적용

    # 3. Bridge & Modules
    vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(device)
    llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(device)
    restorer      = RestormerVolterraFiLM(cond_dim=128).to(device)
    aux_cls       = AuxClassifier(llm_dim=llm_dim).to(device)

    # Trainable params per stage
    if stage == 1:
        for p in llm_model.parameters(): p.requires_grad = False
        params = list(restorer.parameters())
    elif stage == 2:
        for p in restorer.parameters(): p.requires_grad = False
        params = list(vision_to_llm.parameters()) + list(llm_to_cond.parameters()) + list(aux_cls.parameters())
    else:  # Stage 3
        params = list(restorer.parameters()) + list(vision_to_llm.parameters()) + list(llm_to_cond.parameters()) + list(aux_cls.parameters())
        lr = lr * 0.1

    optimizer = torch.optim.Adam(params, lr=lr)
    scaler = GradScaler("cuda")

    # 5. Dataset
    datasets = get_train_datasets(limit=160)
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------- Training Loop ---------------- #
    for epoch in range(1, epochs+1):
        print(f"\n🚀 Stage {stage} Epoch {epoch}/{epochs}")
        pbar = tqdm(dataloader, desc=f"[Stage {stage}] Epoch {epoch}/{epochs}", leave=True)

        for step, (img, gt) in enumerate(pbar, start=1):
            img, gt = img.to(device), gt.to(device)

            # CLIP embeddings
            pil_imgs = [T.ToPILImage()(i.cpu()) for i in img]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                e_clip = clip_model(**inputs).pooler_output  # float32

            # ✅ Projection은 float32, 이후 half 변환
            vision_embed = vision_to_llm(e_clip.to(torch.float32))
            vision_embed = vision_embed.to(dtype=torch.float16)

            # LLM input (prefix + embed)
            prefix = "Distortion type reasoning:"
            prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
            prefix_emb = llm_model.get_input_embeddings()(prefix_ids)
            prefix_emb = prefix_emb.expand(img.size(0), -1, -1).contiguous()
            combined = torch.cat([prefix_emb, vision_embed.unsqueeze(1)], dim=1)

            with autocast("cuda"):
                out = llm_model(inputs_embeds=combined)

                # logits: [B, T, Vocab] → projection 전 hidden
                # 내부 embedding 레이어를 다시 호출해서 hidden으로 변환
                last_token_logits = out.logits[:, -1, :]  # [B, V]
                h_llm = llm_model.get_input_embeddings()(last_token_logits.argmax(dim=-1))


                # z for FiLM
                z = llm_to_cond(h_llm.to(dtype=torch.float32))
                pred = restorer(img, z)

                # Restoration loss
                ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
                psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
                rest_loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

                # Auxiliary classification loss (dummy labels for now)
                type_logits, strength_logits = aux_cls(h_llm)
                fake_type_labels = torch.zeros(img.size(0), dtype=torch.long, device=device)
                fake_strength_labels = torch.zeros(img.size(0), dtype=torch.long, device=device)
                aux_loss = F.cross_entropy(type_logits, fake_type_labels) + \
                           F.cross_entropy(strength_logits, fake_strength_labels)

                loss = rest_loss + lambda_aux * aux_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Explainability 출력
            if step % log_interval == 0:
                try:
                    gen_ids = llm_model.generate(
                        inputs_embeds=combined,
                        max_new_tokens=15,
                        do_sample=False
                    )
                    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    print(f"\n🔎 Explainability: {decoded}")
                except Exception as e:
                    print(f"[Explain log error] {e}")

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rest": f"{rest_loss.item():.4f}",
                "aux": f"{aux_loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val:.4f}"
            })

        # ---- Save ckpt ----
        ckpt_path = os.path.join(save_dir, f"stage{stage}_epoch{epoch}_ssim{ssim_val:.4f}_psnr{psnr_val:.2f}.pth")
        torch.save({
            "epoch": epoch,
            "restorer": restorer.state_dict(),
            "vision_to_llm": vision_to_llm.state_dict(),
            "llm_to_cond": llm_to_cond.state_dict(),
            "aux_cls": aux_cls.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)
        print(f"✓ Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main(stage=1, epochs=2, batch_size=1)
    main(stage=2, epochs=2, batch_size=1)
    main(stage=3, epochs=1, batch_size=1)

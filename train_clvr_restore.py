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

from transformers import (
    CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
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
         save_dir="E:/CLVR_Restore/checkpoints", lambda_aux=0.2,
         log_interval=50, llm_choice="mistral"):
    """
    llm_choice: "opt" | "llama2" | "mistral" | "qwen"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size

    # 2. LLM 선택 (bnb 4bit 적용)
    if llm_choice == "opt":
        model_name = "facebook/opt-125m"
    elif llm_choice == "llama2":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif llm_choice == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif llm_choice == "qwen":
        model_name = "Qwen/Qwen-7B-Chat"
    else:
        raise ValueError(f"Unknown llm_choice: {llm_choice}")

    print(f"⚡ Loading LLM (bnb 4bit): {model_name}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # ✅ Gradient checkpointing (VRAM 절약)
    llm_model.gradient_checkpointing_enable()
    llm_dim = llm_model.config.hidden_size

    # ✅ Stage 2/3에서 LoRA 적용
    if stage in [2, 3]:
        llm_model = apply_lora_to_llm(llm_model)

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

    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = GradScaler("cuda")

    # 5. Dataset
    datasets = get_train_datasets(limit=160, with_labels=True)  # ✅ 라벨 포함
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------- Training Loop ---------------- #
    for epoch in range(1, epochs+1):
        print(f"\n🚀 Stage {stage} Epoch {epoch}/{epochs} | LLM: {llm_choice}")
        pbar = tqdm(dataloader, desc=f"[Stage {stage}] Epoch {epoch}/{epochs}", leave=True)

        for step, batch in enumerate(pbar, start=1):
            img, gt, type_label, strength_label = batch
            img, gt = img.to(device), gt.to(device)
            type_label, strength_label = type_label.to(device), strength_label.to(device)

            # CLIP embeddings
            pil_imgs = [T.ToPILImage()(i.cpu()) for i in img]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                e_clip = clip_model(**inputs).pooler_output
            vision_embed = vision_to_llm(e_clip).to(dtype=torch.float32)

            # LLM input (prompt + embed)
            prefix = "Describe the distortion types and severity (rain, blur, noise, snow):"
            prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
            prefix_emb = llm_model.get_input_embeddings()(prefix_ids)
            prefix_emb = prefix_emb.expand(img.size(0), -1, -1).contiguous()
            combined = torch.cat([prefix_emb, vision_embed.unsqueeze(1)], dim=1)

            with autocast("cuda"):
                out = llm_model(inputs_embeds=combined, output_hidden_states=True)
                h_llm = out.hidden_states[-1][:, -1, :]  # [B, D]

                # z for FiLM
                z = llm_to_cond(h_llm.to(dtype=torch.float32))
                pred = restorer(img, z)

                # Restoration loss
                ssim_val = compute_ssim(pred.clamp(0, 1), gt.clamp(0, 1))
                psnr_val = compute_psnr(pred.clamp(0, 1), gt.clamp(0, 1))
                rest_loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

                # Auxiliary classification loss
                type_logits, strength_logits = aux_cls(h_llm)
                aux_loss = F.cross_entropy(type_logits, type_label) + \
                           F.cross_entropy(strength_logits, strength_label)

                loss = rest_loss + lambda_aux * aux_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Explainability 출력
            if step % log_interval == 0:
                try:
                    gen_ids = llm_model.generate(inputs_embeds=combined, max_new_tokens=20, do_sample=False)
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
    # 디버깅은 opt (소형), 논문 실험은 mistral/llama2/qwen
    main(stage=1, epochs=2, batch_size=1, llm_choice="mistral")
    main(stage=2, epochs=2, batch_size=1, llm_choice="mistral")
    main(stage=3, epochs=1, batch_size=1, llm_choice="mistral")

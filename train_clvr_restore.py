import sys, os, random
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


# ---------------- Prompt Pool ---------------- #
PROMPT_LIST = [
    # --- LLM이 해야 할 작업을 명확히 제시 ---
    "Describe the distortion types and severity (rain, snow, blur, noise, jpeg, haze, low-light, color shift).",
    "Identify the present artifacts and classify each distortion with a severity level.",
    "Which distortions are visible? Indicate type and severity rating.",

    # --- 문장 표현 다양화
    # 같은 의미를 다양한 표현으로 변형 ---
    "List all distortions in the image and mark their intensity (light, medium, heavy).",
    "Detect artifacts: rain / snow / blur / noise / jpeg / haze / low-light / color. Rate severity.",
    "Classify the degradations present in the input and provide a severity label for each.",

    # --- 복합 작업 지시
    # 모델이 다단계 reasoning에 익숙해지도록 유도 ---
    "Task 1: Detect distortion categories (rain, snow, blur, noise, jpeg, haze, low-light, color).\n"
    "Task 2: Rate each distortion from 0 (none) to 2 (strong).",
    "Step 1: Identify all visible distortions.\nStep 2: Assess their severity (low/medium/high).",
    "Perform two subtasks: (a) recognize distortion types, (b) classify severity.",

    # --- 왜곡별 강도 표현 ---
    "Rain or snow intensity: light / medium / heavy. Blur or noise: mild / strong.",
    "For each artifact (rain, snow, blur, noise, jpeg, haze, low-light, color), assign severity: none / low / medium / high.",
    "Classify degradation strength: weak / moderate / severe for each present distortion.",

    # --- 숫자 점수화 ---
    "Rate distortions on a scale of 0 (none), 1 (mild), 2 (moderate), 3 (severe).",
    "Provide a severity score (0–3) for each distortion type detected.",
    "Assign numeric ratings to distortions: rain, snow, blur, noise, jpeg, haze, low-light, color.",

    # --- 주요 왜곡 강조
    # 단순 분류가 아니라 우선순위 reasoning까지 가능하도록 확장 ---
    "Which distortions dominate the image quality? Explain type and severity.",
    "Describe whether rain/snow/blur/noise artifacts are light, medium, or strong.",
    "Give a short reasoning: which distortions affect this image the most and how severe are they?",

    # --- Open-ended reasoning ---
    "Explain the main degradations in the image, listing type and strength.",
    "Identify artifacts and provide both category labels and severity estimates.",
    "What kinds of distortions exist, and how severe is each (low/medium/high)?",

    # --- 🔥 Added: Fixed output format for consistency ---
    "Output format: {distortion: severity, ...}. Example: {rain:2, blur:1, noise:0, jpeg:0, haze:0, low-light:1, color:0}",

    # --- 🔥 Added: Multi-distortion emphasis ---
    "The image may contain multiple distortions simultaneously. Detect and rate each distortion type individually.",

    # --- 🔥 Added: No-distortion case ---
    "If no distortion is visible, output: {all:0}",

    # --- 🔥 Added: Task-driven context ---
    "You are a restoration assistant. Your task is to classify distortions before restoration. "
    "Output format: {distortion: severity}. Example: {rain:3, blur:1, noise:0, jpeg:0, haze:0, low-light:0, color:0}"
]


def get_prompt(mode="random", step=0):
    """Return a prompt string"""
    if mode == "fixed":
        return PROMPT_LIST[0]
    elif mode == "cycle":
        return PROMPT_LIST[step % len(PROMPT_LIST)]
    elif mode == "random":
        return random.choice(PROMPT_LIST)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


# ---------------- Stage3 Finetuning Strategy ---------------- #
def configure_stage3_strategy(restorer, strategy="balanced", base_lr=1e-4):
    """
    Configure which blocks to unfreeze in Stage3
    strategy: "generalization" | "fidelity" | "balanced"
    """
    # 전체 freeze 초기화
    for p in restorer.parameters():
        p.requires_grad = False

    param_groups = []

    if strategy == "generalization":
        # Encoder + Latent
        for m in [restorer.encoder1, restorer.encoder2, restorer.encoder3, restorer.latent]:
            for p in m.parameters():
                p.requires_grad = True
        param_groups.append({"params": restorer.encoder1.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.encoder2.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.encoder3.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.latent.parameters(), "lr": base_lr})

    elif strategy == "fidelity":
        # Decoder + Refinement
        for m in [restorer.decoder1, restorer.decoder2, restorer.decoder3, restorer.refinement]:
            for p in m.parameters():
                p.requires_grad = True
        param_groups.append({"params": restorer.decoder1.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.decoder2.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.decoder3.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.refinement.parameters(), "lr": base_lr})

    elif strategy == "balanced":
        # Encoder + Decoder full, Latent 50%, Refinement full
        for m in [restorer.encoder1, restorer.encoder2, restorer.encoder3,
                  restorer.decoder1, restorer.decoder2, restorer.decoder3,
                  restorer.refinement]:
            for p in m.parameters():
                p.requires_grad = True
        # latent 절반 freeze
        latent_params = list(restorer.latent.parameters())
        n = len(latent_params)
        for i, p in enumerate(latent_params):
            if i < n // 2:  # 앞 절반 freeze
                p.requires_grad = False
            else:
                p.requires_grad = True

        param_groups.append({"params": restorer.encoder1.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.decoder1.parameters(), "lr": base_lr})
        param_groups.append({"params": restorer.refinement.parameters(), "lr": base_lr})
        param_groups.append({"params": [p for p in latent_params if p.requires_grad], "lr": base_lr * 0.5})

    else:
        raise ValueError(f"Unknown Stage3 strategy: {strategy}")

    return param_groups


# ---------------- Main Training ---------------- #
def main(stage=1, epochs=5, batch_size=1, lr=1e-4,
         save_dir="E:/CLVR_Restore/checkpoints", lambda_aux=0.2,
         log_interval=50, llm_choice="mistral", prompt_mode="random",
         stage3_strategy="balanced"):
    """
    llm_choice: "opt" | "llama2" | "mistral" | "qwen"
    prompt_mode: "fixed" | "random" | "cycle"
    stage3_strategy: "generalization" | "fidelity" | "balanced"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    ### 1. CLIP Vision Encoder
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_dim = clip_model.config.hidden_size

    # LLM 선택 (bnb 4bit 적용)
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

    ### 3. LLM with LoRA
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    llm_model.gradient_checkpointing_enable()
    llm_dim = llm_model.config.hidden_size

    if stage in [2, 3]:
        llm_model = apply_lora_to_llm(llm_model)

    ### 2. VisionToLLMAdapter
    vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(device)
    ### 4. LLMToCondition (MLP → z)
    llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(device)
    ### 5. FiLM Injection into ReVolT Backbone
    restorer      = RestormerVolterraFiLM(cond_dim=128).to(device)
    aux_cls       = AuxClassifier(llm_dim=llm_dim).to(device)

    if stage == 1:
        for p in llm_model.parameters(): p.requires_grad = False
        params = list(restorer.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

    elif stage == 2:
        for p in restorer.parameters(): p.requires_grad = False
        params = list(vision_to_llm.parameters()) + list(llm_to_cond.parameters()) + list(aux_cls.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

    else:  # Stage 3
        param_groups = configure_stage3_strategy(restorer, strategy=stage3_strategy, base_lr=lr*0.1)

        # vision adapter, condition head, aux classifier는 항상 학습 
        for p in vision_to_llm.parameters(): p.requires_grad = True
        for p in llm_to_cond.parameters(): p.requires_grad = True
        for p in aux_cls.parameters(): p.requires_grad = True
        param_groups.append({"params": vision_to_llm.parameters(), "lr": lr})
        param_groups.append({"params": llm_to_cond.parameters(), "lr": lr})
        param_groups.append({"params": aux_cls.parameters(), "lr": lr})

        optimizer = torch.optim.AdamW(param_groups)

    scaler = GradScaler("cuda")

    datasets = get_train_datasets(limit=160, with_labels=True)
    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------- Training Loop ---------------- #
    for epoch in range(1, epochs+1):
        print(f"\n🚀 Stage {stage} Epoch {epoch}/{epochs} | LLM: {llm_choice} | Prompt mode: {prompt_mode} | Strategy: {stage3_strategy}")
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

            # ✅ Prompt 다양화: 문자열 → 토큰화 → 임베딩
            prompt_str = get_prompt(mode=prompt_mode, step=step)
            prefix_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
            prefix_emb = llm_model.get_input_embeddings()(prefix_ids)
            prefix_emb = prefix_emb.expand(img.size(0), -1, -1).contiguous()

            # vision_embed 붙이기
            combined = torch.cat([prefix_emb, vision_embed.unsqueeze(1)], dim=1)

            # ✅ attention_mask도 길이를 맞춰줌
            attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)


            with autocast("cuda"):
                out = llm_model(inputs_embeds=combined, output_hidden_states=True)
                h_llm = out.hidden_states[-1][:, -1, :]  # [B, D]

                ### 4. LLMToCondition (MLP → z)
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
                    with torch.amp.autocast("cuda", enabled=False):
                        gen_ids = llm_model.generate(
                            inputs_embeds=combined.to(dtype=llm_model.dtype),
                            attention_mask=attn_mask,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
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

        ckpt_path = os.path.join(save_dir, f"stage{stage}_epoch{epoch}_mode{prompt_mode}_strat{stage3_strategy}_ssim{ssim_val:.4f}_psnr{psnr_val:.2f}.pth")
        torch.save({
            "epoch": epoch,
            "restorer": restorer.state_dict(),
            "vision_to_llm": vision_to_llm.state_dict(),
            "llm_to_cond": llm_to_cond.state_dict(),
            "aux_cls": aux_cls.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)
        print(f"✓ Saved checkpoint: {ckpt_path}")


# ---------------- Ablation Runner ---------------- #
if __name__ == "__main__":
    # Ablation: prompt mode × Stage3 전략
    for mode in ["fixed", "random", "cycle"]:
        for strat in ["generalization", "fidelity", "balanced"]:
            print(f"\n\n===== 🔬 Running Full Experiment with prompt mode: {mode}, strategy: {strat} =====\n")
            
            # Stage1: backbone 학습 (20 epoch)
            main(stage=1, epochs=20, batch_size=1,
                 llm_choice="mistral", prompt_mode=mode,
                 stage3_strategy=strat)

            # Stage2: adapter 학습 (10 epoch)
            main(stage=2, epochs=10, batch_size=1,
                 llm_choice="mistral", prompt_mode=mode,
                 stage3_strategy=strat)

            # Stage3: joint fine-tuning (10 epoch)
            main(stage=3, epochs=10, batch_size=1,
                 llm_choice="mistral", prompt_mode=mode,
                 stage3_strategy=strat)

# LLM (예: Mistral-7B) 이 get_prompt() 로 전달받은 프롬프트에 대해 추론(Generate) 한 결과
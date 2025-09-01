# file: test_debug_clvr_restore.py
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from transformers import CLIPModel, CLIPProcessor, AutoModel

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition


# ---------------------------
# Config
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = r"E:/CLVR_Restore/checkpoints/epoch_1_ssim0.1064_psnr10.83.pth"
IMG_PATH  = r"E:/CLVR_Restore/data/CSD/Test/Snow/4.tif"
GT_PATH   = r"E:/CLVR_Restore/data/CSD/Test/Gt/4.tif"   # ✅ GT 경로 있으면 평가 가능
SAVE_PATH = r"E:/CLVR_Restore/restored_debug.png"


# ---------------------------
# Load Models
# ---------------------------
print("🔹 Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(DEVICE)
clip_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_dim = clip_model.config.hidden_size  # 768

print("🔹 Loading LLM (Mistral 7B)...")
llm_model = AutoModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto",
)
llm_model.eval()
llm_dim = llm_model.config.hidden_size  # 4096

print("🔹 Loading Bridge + Restorer...")
vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(DEVICE)
llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(DEVICE)
restorer      = RestormerVolterraFiLM(cond_dim=128).to(DEVICE)


# ---------------------------
# Load Checkpoint
# ---------------------------
print(f"🔹 Loading checkpoint: {CKPT_PATH}")
state = torch.load(CKPT_PATH, map_location=DEVICE)
restorer.load_state_dict(state["restorer"])
vision_to_llm.load_state_dict(state["vision_to_llm"])
llm_to_cond.load_state_dict(state["llm_to_cond"])
restorer.eval(); vision_to_llm.eval(); llm_to_cond.eval()


# ---------------------------
# Input Image
# ---------------------------
print(f"🔹 Loading test image: {IMG_PATH}")
img = Image.open(IMG_PATH).convert("RGB")
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# GT (있을 때만 평가)
gt_tensor = None
if os.path.exists(GT_PATH):
    gt_img = Image.open(GT_PATH).convert("RGB")
    gt_tensor = transform(gt_img).unsqueeze(0).to(DEVICE)


# ---------------------------
# Pipeline Debug
# ---------------------------

# (1) CLIP
inputs = processor(images=img, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    e_clip = clip_model(**inputs).pooler_output
print("✅ CLIP embedding:", e_clip.shape)  # [B, 768]

# (2) Vision→LLM
vision_embed = vision_to_llm(e_clip).to(dtype=torch.float16)
print("✅ Vision→LLM embedding:", vision_embed.shape)  # [B, 4096]

# (3) LLM Forward
with torch.no_grad():
    llm_out = llm_model(inputs_embeds=vision_embed.unsqueeze(1))
    h_llm = llm_out.last_hidden_state[:, -1, :]
print("✅ LLM hidden:", h_llm.shape)  # [B, 4096]

# (4) LLM→Condition
z = llm_to_cond(h_llm.to(dtype=torch.float32))
print("✅ Condition vector z:", z.shape)  # [B, 128]
print("🔎 First 10 values of z:", z[0, :10].detach().cpu().numpy())  # 일부 값 출력

# (5) Restorer
with torch.no_grad():
    pred = restorer(img_tensor, z)
print("✅ Restored image:", pred.shape)  # [B, 3, 256, 256]


# ---------------------------
# Save Restored Output
# ---------------------------
pred_img = pred[0].detach().cpu().clamp(0, 1)
T.ToPILImage()(pred_img).save(SAVE_PATH)
print(f"💾 Saved restored image to {SAVE_PATH}")


# ---------------------------
# Optional: Metrics
# ---------------------------
if gt_tensor is not None:
    pred_np = pred_img.permute(1, 2, 0).numpy()
    gt_np   = gt_tensor[0].permute(1, 2, 0).cpu().numpy()

    psnr_val = compute_psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = compute_ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)

    print(f"📊 Metrics → PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
else:
    print("⚠️ GT image not found, skipping PSNR/SSIM.")

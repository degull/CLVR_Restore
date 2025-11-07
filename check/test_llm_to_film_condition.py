# ---------------------------------------------------------
# file: check/test_llm_to_film_condition.py
# ---------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.llm_lora import apply_lora_to_llm
from models.llm_condition_head import LLMToCondition
from models.restormer_volterra_film import RestormerVolterraFiLM


# ---------------------------------------------------------
# 1️⃣ LLM 로드 (Mistral-7B + LoRA)
# ---------------------------------------------------------
def load_mistral_llm(model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
    print(f"🚀 Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    llm_model = apply_lora_to_llm(llm_model)
    llm_model.eval()
    return llm_model, tokenizer


# ---------------------------------------------------------
# 2️⃣ LLM hidden → FiLM conditioning vector
# ---------------------------------------------------------
def llm_to_condition_vector(llm_model, tokenizer, prompt, device="cuda"):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = llm_model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

    # LLM → FiLM conditioning
    cond_head = LLMToCondition(llm_dim=last_hidden.shape[-1], cond_dim=128).to(device).to(last_hidden.dtype)
    z = cond_head(last_hidden)
    return z, last_hidden


# ---------------------------------------------------------
# 3️⃣ Restoration Network (Restormer+Volterra+FiLM)
# ---------------------------------------------------------
def run_restoration_with_condition(image_path, z, device="cuda"):
    print(f"🖼️ Loading degraded image: {image_path}")
    image = Image.open(image_path).convert("RGB")  # ✅ RGB 고정

    tfm = transforms.Compose([transforms.ToTensor()])
    x = tfm(image).unsqueeze(0).to(device)
    x = x * 2 - 1  # [0,1] → [-1,1]

    model = RestormerVolterraFiLM(cond_dim=128).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x, z.float())

    # ✅ 안정화: [-1,1] 범위 내에서 tanh로 정규화
    out = torch.tanh(out)
    out = (out + 1) / 2

    restored = out.squeeze(0).clamp(0, 1).cpu()
    out_img = transforms.ToPILImage()(restored)

    save_path = "E:/CLVR_Restore/check/restored_with_condition.png"
    out_img.save(save_path)
    print(f"✅ Restored image saved at: {save_path}")
    print(f"Output mean={restored.mean():.4f}, min={restored.min():.4f}, max={restored.max():.4f}")



# ---------------------------------------------------------
# 4️⃣ 통합 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Load LLM
    llm_model, tokenizer = load_mistral_llm(device=device)

    # Step 2: Prompt reasoning
    prompt = (
        "You are a vision-language restoration expert. "
        "The input image appears snowy with white granular distortions. "
        "Describe how to restore it by reasoning about the distortion and suggesting FiLM modulation hints."
    )

    z, h_llm = llm_to_condition_vector(llm_model, tokenizer, prompt, device=device)
    print(f"✅ Condition vector generated: shape={tuple(z.shape)}, mean={z.mean():.4f}, std={z.std():.4f}")

    # Step 3: Apply FiLM-conditioned restoration
    img_path = "E:/CLVR_Restore/data/CSD/Test/Snow/4.tif"
    run_restoration_with_condition(img_path, z, device=device)

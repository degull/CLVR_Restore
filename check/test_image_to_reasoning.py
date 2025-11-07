# file: check/test_image_to_reasoning.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.llm_lora import apply_lora_to_llm
from models.vision_llm_adapter import VisionToLLMAdapter
from models.run_clip_film_restore import CLIPAdapter

# ---------------------------------------------------------
# 1️⃣ Mistral LLM + LoRA 로드
# ---------------------------------------------------------
def load_llm(model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
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
# 2️⃣ 이미지 기반 Reasoning 생성
# ---------------------------------------------------------
def image_to_reasoning(image_path, device="cuda"):
    # Step 1. Load CLIP encoder
    clip_adapter = CLIPAdapter(cond_dim=128, device=device)
    vision_to_llm = VisionToLLMAdapter(clip_dim=768, llm_dim=4096).to(device)

    # Step 2. Load LLM
    llm, tokenizer = load_llm(device=device)

    # Step 3. Load image
    image = Image.open(image_path).convert("RGB")
    print(f"🖼️ Loaded image: {image_path}")

    # Step 4. CLIP → embedding → project to LLM
    with torch.no_grad():
        e_clip = clip_adapter.clip_model.vision_model(**clip_adapter.processor(images=image, return_tensors="pt").to(device))
        pooled = e_clip.pooler_output  # [1,768]
        h_llm = vision_to_llm(pooled)  # [1,4096]

    # Step 5. Prompt
    base_prompt = (
        "You are a vision-language restoration expert. "
        "Given the visual features of an image, analyze what type of distortion it likely has "
        "(rain, snow, blur, noise, JPEG, etc.) and explain step-by-step how to restore it."
    )

    # Step 6. Generate text
    inputs = tokenizer(base_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = llm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("\n--- 🧠 LLM Reasoning Output ---\n")
    print(text)
    print("\n✅ Reasoning generation complete.\n")

# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = "E:/CLVR_Restore/data/CSD/Test/Snow/4.tif"
    image_to_reasoning(image_path, device=device)

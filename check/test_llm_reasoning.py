# file: check/test_llm_reasoning.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.llm_lora import apply_lora_to_llm
from models.aux_classifier import AuxClassifier

# ---------------------------------------------------------
# 1️⃣ Load Base LLM (Mistral-7B)
# ---------------------------------------------------------
def load_mistral_llm(model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
    print(f"🚀 Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    # Apply LoRA adapters
    llm_model = apply_lora_to_llm(llm_model)
    llm_model.eval()
    return llm_model, tokenizer

# ---------------------------------------------------------
# 2️⃣ Reasoning & Classification Test
# ---------------------------------------------------------
def test_llm_reasoning():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model, tokenizer = load_mistral_llm(device=device)

    # Aux classifier (for distortion type & strength)
    aux_cls = AuxClassifier(llm_dim=4096).to(device).to(model.dtype)

    # Prompt example (like Stage-2 prompt)
    prompt = (
        "You are an expert in image restoration. "
        "Describe the type and severity of distortion present in this image, "
        "and reason step-by-step about how to restore it."
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run forward
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        type_logits, strength_logits = aux_cls(last_hidden)

    # Decode reasoning
    generated = model.generate(
        **inputs, max_new_tokens=100, do_sample=True, temperature=0.7
    )
    reasoning_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Print results
    print("\n✅ LLM Reasoning Forward 성공")
    print(f"Hidden shape: {tuple(last_hidden.shape)}")
    print(f"Type logits: {type_logits.detach().cpu().numpy().round(3)}")
    print(f"Strength logits: {strength_logits.detach().cpu().numpy().round(3)}")
    print("\n--- Reasoning Output ---\n")
    print(reasoning_text)

# ---------------------------------------------------------
if __name__ == "__main__":
    test_llm_reasoning()

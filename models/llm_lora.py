# models/llm_lora.py
from peft import LoraConfig, get_peft_model

def apply_lora_to_llm(llm_model, r=8, alpha=16, dropout=0.05, target_modules=None):
    """
    HuggingFace LLM 모델에 LoRA 어댑터를 붙여줌.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 기본 어텐션 프로젝션

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm_model = get_peft_model(llm_model, lora_config)
    return llm_model

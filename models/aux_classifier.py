# models/aux_classifier.py
import torch.nn as nn
import torch

class AuxClassifier(nn.Module):
    """
    Distortion type & strength 분류 보조 classifier
    - 입력: LLM hidden (dim)
    - 출력: type logits, strength logits
    """
    def __init__(self, llm_dim=4096, num_types=5, num_strengths=3):
        super().__init__()
        self.type_head = nn.Linear(llm_dim, num_types)
        self.strength_head = nn.Linear(llm_dim, num_strengths)

    def forward(self, h_llm):
        # h_llm: [B, D]
        type_logits = self.type_head(h_llm)
        strength_logits = self.strength_head(h_llm)
        return type_logits, strength_logits

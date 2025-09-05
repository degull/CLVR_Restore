import torch
import torch.nn as nn

class VisionToLLMAdapter(nn.Module):
    """
    CLIP Vision Encoder embedding -> LLM hidden dimension으로 projection
    """
    def __init__(self, clip_dim=768, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(clip_dim, llm_dim)

    def forward(self, e_clip):
        # e_clip: [B, clip_dim]
        return self.proj(e_clip)   # [B, llm_dim]

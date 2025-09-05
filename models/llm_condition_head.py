import torch
import torch.nn as nn

class LLMToCondition(nn.Module):
    """
    LLM hidden state -> FiLM condition vector z
    """
    def __init__(self, llm_dim=4096, cond_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cond_dim)
        )

    def forward(self, h_llm):
        # h_llm: [B, llm_dim]
        return self.fc(h_llm)  # [B, cond_dim]

# file: run_clip_film_restore.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torchvision import transforms

from models.restormer_volterra_film import RestormerVolterraFiLM


# ---------------- CLIP Adapter ---------------- #
class CLIPAdapter(nn.Module):
    """ CLIP Vision Encoder → condition vector z """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", cond_dim=128, device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP (학습 안 함)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # projection: hidden_size → cond_dim
        hidden_dim = self.clip_model.config.vision_config.hidden_size
        self.proj = nn.Linear(hidden_dim, cond_dim).to(device)

    def forward(self, image: Image.Image):
        # image: PIL.Image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output  # [B, hidden_dim]

        z = self.proj(pooled)  # [B, cond_dim]
        return z


# ---------------- End-to-End Restore Pipeline ---------------- #
class CLVRRestore(nn.Module):
    def __init__(self, cond_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.clip_adapter = CLIPAdapter(cond_dim=cond_dim, device=device)
        self.restorer = RestormerVolterraFiLM(cond_dim=cond_dim).to(device)

    def forward(self, image: Image.Image):
        """ image: PIL.Image, returns restored tensor """
        # 1. condition vector z from CLIP
        z = self.clip_adapter(image)

        # 2. 원본 해상도로 tensor 변환
        tfm = transforms.Compose([transforms.ToTensor()])
        x = tfm(image).unsqueeze(0).to(self.device)   # [1, 3, H, W]

        # 3. restoration
        with torch.no_grad():
            out = self.restorer(x, z)

        return out


# ---------------- Example Run ---------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = CLVRRestore(cond_dim=128, device=device)

    img = Image.open("E:/CLVR_Restore/data/CSD/Test/Snow/4.tif").convert("RGB")

    restored = pipeline(img)  # [1, 3, H, W]
    print("Restored:", restored.shape)

    import torchvision.transforms.functional as TF
    TF.to_pil_image(restored.squeeze(0).cpu().clamp(0,1)).save("restored.png")
    print("Saved to restored.png")

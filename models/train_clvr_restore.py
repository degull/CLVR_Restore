# 학습 루프 예시 (dummy dataset 기준, 실제는 Rain100H 등 연결)
# train_clvr_restore.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, LlamaModel
from skimage.metrics import structural_similarity as ssim_metric  # ✅ 실제 SSIM

from models.restormer_volterra_film import RestormerVolterraFiLM
from models.vision_llm_adapter import VisionToLLMAdapter
from models.llm_condition_head import LLMToCondition


# ✅ 데이터셋 예시 (dummy dataset)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10, size=256):
        self.n = n
        self.size = size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.rand(3, self.size, self.size)
        y = x.clone()  # dummy GT
        return x, y


def compute_ssim(pred, gt):
    """ pred, gt: [B,3,H,W], 0~1 range """
    pred = pred.detach().cpu().numpy().transpose(0,2,3,1)
    gt   = gt.detach().cpu().numpy().transpose(0,2,3,1)
    ssim_vals = []
    for i in range(pred.shape[0]):
        ssim_vals.append(ssim_metric(gt[i], pred[i], channel_axis=2, data_range=1.0))
    return torch.tensor(sum(ssim_vals)/len(ssim_vals))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # 1. Pretrained CLIP vision encoder
    # ---------------------------
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.to(device)
    clip_model.eval()
    clip_dim = clip_model.config.hidden_size  # =768

    # ---------------------------
    # 2. LLM (주의: LLaMA-2는 HuggingFace 계정/허가 필요)
    # ---------------------------
    llm_model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    for p in llm_model.parameters():
        p.requires_grad = False  # LoRA 적용 시 일부 unfrozen

    llm_dim = llm_model.config.hidden_size  # =4096

    # ---------------------------
    # 3. Bridge modules
    # ---------------------------
    vision_to_llm = VisionToLLMAdapter(clip_dim=clip_dim, llm_dim=llm_dim).to(device)
    llm_to_cond   = LLMToCondition(llm_dim=llm_dim, cond_dim=128).to(device)

    # ---------------------------
    # 4. Restorer backbone
    # ---------------------------
    restorer = RestormerVolterraFiLM(cond_dim=128).to(device)

    optimizer = torch.optim.Adam(
        list(vision_to_llm.parameters()) +
        list(llm_to_cond.parameters()) +
        list(restorer.parameters()),
        lr=1e-4
    )

    # ---------------------------
    # 5. Dummy dataloader
    # ---------------------------
    dataloader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)

    for epoch in range(2):
        for img, gt in dataloader:
            img, gt = img.to(device), gt.to(device)

            # (1) CLIP vision embedding
            with torch.no_grad():
                e_clip = clip_model(img).pooler_output  # [B, 768]

            # (2) Vision->LLM
            vision_embed = vision_to_llm(e_clip)       # [B, 4096]

            # (3) LLM forward (단순 hidden 사용)
            llm_out = llm_model(inputs_embeds=vision_embed.unsqueeze(1))
            h_llm = llm_out.last_hidden_state[:, -1, :]  # [B, 4096]

            # (4) LLM->Condition z
            z = llm_to_cond(h_llm)                     # [B, 128]

            # (5) Restore
            pred = restorer(img, z)

            # (6) Loss
            ssim_val = compute_ssim(pred.clamp(0,1), gt.clamp(0,1))
            loss = F.l1_loss(pred, gt) + (1 - ssim_val) * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss {loss.item():.4f}, SSIM {ssim_val:.4f}")

if __name__ == "__main__":
    main()

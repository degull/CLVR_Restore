# restormer_volterra_film.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volterra_layer import VolterraLayer2D

import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
#  FiLM Modules
# ============================================================
class FiLM(nn.Module):
    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta


class FiLMGenerator(nn.Module):
    """Condition vector → (gamma, beta) 생성기"""
    def __init__(self, cond_dim, feat_dim, scale=0.05):  # ✅ 안정화 스케일 추가
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)
        self.scale = scale

    def forward(self, z):
        gamma = self.to_gamma(z) * self.scale
        beta = self.to_beta(z) * self.scale
        return gamma, beta


class FiLMLayer(nn.Module):
    def __init__(self, feat_dim, cond_dim):
        super().__init__()
        self.gen = FiLMGenerator(cond_dim, feat_dim)
        self.film = FiLM()

    def forward(self, x, z):
        gamma, beta = self.gen(z)
        return self.film(x, gamma, beta)


# ============================================================
#  Layer Normalization
# ============================================================
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + 1e-5)
        return self.weight * x


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return self.weight * x + self.bias


def LayerNorm(normalized_shape, bias=False):
    return WithBiasLayerNorm(normalized_shape) if bias else BiasFreeLayerNorm(normalized_shape)


# ============================================================
#  Gated-Dconv FeedForward Network (GDFN)
# ============================================================
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ============================================================
#  Multi-Dconv Head Transposed Attention (MDTA)
# ============================================================
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


# ============================================================
#  Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type, volterra_rank, cond_dim=128):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.film1 = FiLMLayer(dim, cond_dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.volterra1 = VolterraLayer2D(dim, dim, rank=volterra_rank)

        self.norm2 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.film2 = FiLMLayer(dim, cond_dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.volterra2 = VolterraLayer2D(dim, dim, rank=volterra_rank)

    def forward(self, x, z):
        x = x + self.volterra1(self.attn(self.film1(self.norm1(x), z)))
        x = x + self.volterra2(self.ffn(self.film2(self.norm2(x), z)))
        return x


# ============================================================
#  Encoder / Decoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x, z):
        for blk in self.blocks:
            x = blk(x, z)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x, z):
        for blk in self.blocks:
            x = blk(x, z)
        return x


# ============================================================
#  Downsample / Upsample
# ============================================================
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# ============================================================
#  Restormer + Volterra + FiLM Main Network
# ============================================================
class RestormerVolterraFiLM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias',
                 volterra_rank=4, cond_dim=128):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.encoder1 = Encoder(dim, num_blocks[0], num_heads=heads[0],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)
        self.down1 = Downsample(dim)

        self.encoder2 = Encoder(dim * 2, num_blocks[1], num_heads=heads[1],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)
        self.down2 = Downsample(dim * 2)

        self.encoder3 = Encoder(dim * 4, num_blocks[2], num_heads=heads[2],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)
        self.down3 = Downsample(dim * 4)

        self.latent = Encoder(dim * 8, num_blocks[3], num_heads=heads[3],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type,
                              volterra_rank=volterra_rank, cond_dim=cond_dim)

        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = Decoder(dim * 4, num_blocks[2], num_heads=heads[2],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = Decoder(dim * 2, num_blocks[1], num_heads=heads[1],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = Decoder(dim, num_blocks[0], num_heads=heads[0],
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type,
                                volterra_rank=volterra_rank, cond_dim=cond_dim)

        self.refinement = Encoder(dim, num_refinement_blocks, num_heads=heads[0],
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type,
                                  volterra_rank=volterra_rank, cond_dim=cond_dim)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def _pad_and_add(self, up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:],
                                      mode='bilinear', align_corners=False)
        return up_tensor + skip_tensor

    def forward(self, x, z):
        # ✅ 입력 정규화 [0,1] → [-1,1]
        x = x * 2 - 1

        # ✅ CLIP z normalization (zero-mean, unit-var)
        z = (z - z.mean(dim=-1, keepdim=True)) / (z.std(dim=-1, keepdim=True) + 1e-6)

        x1 = self.patch_embed(x)
        x2 = self.encoder1(x1, z)
        x3 = self.encoder2(self.down1(x2), z)
        x4 = self.encoder3(self.down2(x3), z)
        x5 = self.latent(self.down3(x4), z)
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4), z)
        x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3), z)
        x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2), z)
        x9 = self.refinement(x8, z)
        out = self.output(x9 + x1)

        # ✅ 안정화 및 출력 복원 [-1,1] → [0,1]
        out = torch.tanh(out)
        out = (out + 1) / 2
        return out


# ============================================================
#  Sanity Test
# ============================================================
if __name__ == '__main__':
    model = RestormerVolterraFiLM()
    dummy_img = torch.rand(1, 3, 256, 256)
    dummy_z = torch.randn(1, 128)
    out = model(dummy_img, dummy_z)
    print("Output shape:", out.shape)
    print("Output range:", out.min().item(), "→", out.max().item())

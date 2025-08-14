import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

@torch.no_grad()
def _prob_binary(logits):
    if logits.dim()==2 and logits.size(1)==1:
        logits = logits.squeeze(1)
    p = torch.sigmoid(logits)          # (B,)
    return torch.stack([p, 1-p], dim=1)  # (B,2)

@torch.no_grad()
def _prob_multiclass(logits):
    return F.softmax(logits, dim=1)    # (B,K)

@torch.no_grad()
def _kl(p, q, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return (p * (p.log() - q.log())).sum(dim=1)  # (B,)

class ConsistencyDetector(nn.Module):
    """
    Lp 공격 감지 (임계 0.5 고정 가정).
    - 여러 변환 기반 KL 벡터를 뽑고, 2층 MLP 헤드로 로짓 출력.
    - BN/z-score를 쓰지 않아 배치 의존성이 낮고 0.5 임계와의 어긋남이 줄어듦.
    """
    def __init__(self, victim_model,
                 blur_sigmas=(0.8, 1.2),
                 bit_depths=(3, 4, 5),
                 down_up_scales=(0.75, 0.9),
                 gauss_noise_stds=(0.01, 0.02),
                 temp=1.0,
                 hidden=16, drop=0.2):
        super().__init__()
        self.model = victim_model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.blur_sigmas = blur_sigmas
        self.bit_depths = bit_depths
        self.down_up_scales = down_up_scales
        self.gauss_noise_stds = gauss_noise_stds
        self.temp = temp

        T = len(blur_sigmas) + len(bit_depths) + len(down_up_scales) + len(gauss_noise_stds)
        self.T = T

        # 2층 MLP 헤드 (BN 없음, 임계 0.5에 맞춰 학습만으로 분리)
        self.head = nn.Sequential(
            nn.Linear(T, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 1)   # logits
        )

    @torch.no_grad()
    def _probs(self, x):
        logits = self.model(x)
        if logits.dim()==2 and logits.size(1)==1:
            return _prob_binary(logits)
        else:
            return _prob_multiclass(logits)

    @torch.no_grad()
    def _bitsqueeze(self, x, bits):
        x01 = (x + 1) / 2
        levels = 2 ** bits
        xq = torch.round(x01 * (levels - 1)) / (levels - 1)
        return xq * 2 - 1

    @torch.no_grad()
    def _down_up(self, x, scale):
        B, C, H, W = x.shape
        h2 = max(1, int(H * scale))
        w2 = max(1, int(W * scale))
        xd = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=False)
        return F.interpolate(xd, size=(H, W), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _blur(self, x, sigma):
        k = 3 if sigma <= 1.0 else 5
        return TF.gaussian_blur(x, kernel_size=k, sigma=sigma)

    @torch.no_grad()
    def _add_gauss_noise(self, x, std):
        return torch.clamp(x + torch.randn_like(x)*std, -1.0, 1.0)

    def forward(self, x):
        with torch.no_grad():
            p0 = self._probs(x / self.temp)
            feats = []

            for s in self.blur_sigmas:
                pb = self._probs(self._blur(x, s) / self.temp)
                feats.append(_kl(p0, pb))

            for b in self.bit_depths:
                ps = self._probs(self._bitsqueeze(x, b) / self.temp)
                feats.append(_kl(p0, ps))

            for sc in self.down_up_scales:
                pu = self._probs(self._down_up(x, sc) / self.temp)
                feats.append(_kl(p0, pu))

            for std in self.gauss_noise_stds:
                pn = self._probs(self._add_gauss_noise(x, std) / self.temp)
                feats.append(_kl(p0, pn))

            score_vec = torch.stack(feats, dim=1)  # (B, T)

            # 간단한 안정화: 로그 스케일(양수 보장), 너무 큰 값 클립
            score_vec = torch.log1p(torch.clamp(score_vec, min=0.0))
            score_vec = torch.clamp(score_vec, max=6.0)

        return self.head(score_vec)  # (B,1) logits

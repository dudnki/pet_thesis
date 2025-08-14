import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- confidence helpers ----
def _binary_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,1) 또는 (B,)
    반환: (B,)  — 예측 클래스에 대한 확신도 max(p, 1-p)
    """
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    p = torch.sigmoid(logits)                  # (B,)
    conf = torch.maximum(p, 1 - p)             # (B,)
    return conf

def _multiclass_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,K)
    반환: (B,) — argmax 클래스 확률
    """
    probs = F.softmax(logits, dim=1)           # (B,K)
    pred = probs.argmax(1)                     # (B,)
    return probs[torch.arange(logits.size(0), device=logits.device), pred]


class PatchZeroDetector(nn.Module):
    """
    PatchZero-style 감지기 (학습 가능 형태).
    - victim_model: 원 분류 모델(efficientnet_b0 등, eval/로짓 출력)
    - 슬라이딩 윈도우 occlusion으로 '가리면 복구되는 만큼'을 score로 사용
    - score를 얇은 선형층(logit = a*score + b)으로 보정하여 BCEWithLogitsLoss에 바로 사용
    """
    def __init__(self, victim_model,
                 win=64, stride=32, pad=16, topk=1, expand=8, fill="mean"):
        super().__init__()
        self.model = victim_model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.win = win
        self.stride = stride
        self.pad = pad
        self.topk = topk
        self.expand = expand
        self.fill = fill  # "zero" or "mean"

        # 얇은 보정층: score -> logit  (학습되는 파라미터)
        self.scale = nn.Parameter(torch.tensor(10.0))  # 초기 기울기
        self.bias  = nn.Parameter(torch.tensor(0.0))   # 초기 절편

    @torch.no_grad()
    def _score_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        각 이미지별로, 윈도우를 가렸을 때 '복구 이득(recovery gain)'의 상위 top-k 평균 점수 반환.
        return: score (B,)
        """
        device = x.device
        B, C, H, W = x.shape

        logits_base = self.model(x)  # (B,1) 또는 (B,K)

        # 이진/다중 분류 자동 분기
        if logits_base.dim() == 2 and logits_base.size(1) == 1:
            base_conf = _binary_confidence(logits_base)   # (B,)
            is_binary = True
        else:
            base_conf = _multiclass_confidence(logits_base)  # (B,)
            is_binary = False

        # padding
        if self.pad > 0:
            x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        else:
            x_pad = x

        Hp, Wp = x_pad.shape[-2:]
        win, stride = self.win, self.stride
        h_steps = max(1, (Hp - win) // stride + 1)
        w_steps = max(1, (Wp - win) // stride + 1)

        # fill용 mean 값
        if self.fill == "mean":
            mean_vals = x_pad.view(B, C, -1).mean(-1).view(B, C, 1, 1)
        else:
            mean_vals = None

        gains_all = []  # (num_windows, B)
        for ih in range(h_steps):
            for iw in range(w_steps):
                y0 = ih * stride
                x0 = iw * stride
                x_occ = x_pad.clone()
                if self.fill == "zero":
                    x_occ[:, :, y0:y0+win, x0:x0+win] = 0
                else:
                    x_occ[:, :, y0:y0+win, x0:x0+win] = mean_vals

                if self.pad > 0:
                    x_occ = x_occ[:, :, self.pad:self.pad+H, self.pad:self.pad+W]

                logits_occ = self.model(x_occ)
                if is_binary:
                    conf_occ = _binary_confidence(logits_occ)       # (B,)
                else:
                    conf_occ = _multiclass_confidence(logits_occ)   # (B,)
                gains_all.append(torch.abs(conf_occ - base_conf).unsqueeze(0))  # (1,B)

        if len(gains_all) == 0:
            # 윈도우가 없으면 점수 0
            return torch.zeros(B, device=device)

        gains = torch.cat(gains_all, dim=0)  # (num_windows, B)
        k = min(self.topk, gains.size(0))
        topk_vals, _ = gains.topk(k=k, dim=0)
        score = topk_vals.mean(dim=0)        # (B,)
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        출력: logits (B,1) — BCEWithLogitsLoss에 바로 사용.
        """
        with torch.no_grad():
            score = self._score_windows(x)   # (B,)

        logit = self.scale * score + self.bias  # (B,)
        return logit.unsqueeze(1)               # (B,1)

import torch
import torch.nn as nn

def cw_attack(model, images, targets, c=1.0, iters=500, lr=0.01, device='cuda'):
    """
    BCE 기반 Carlini & Wagner L2 공격 (정규화 이미지 기준)
    Args:
        model: 학습된 모델 (logit 출력, sigmoid 적용 X)
        images: 정규화된 입력 이미지 텐서 [B, C, H, W]
        targets: float형 라벨 [B, 1], 값은 0 또는 1
        c: loss weight (perturbation vs. misclassification)
        iters: 최적화 반복 횟수
        lr: learning rate
    Returns:
        adversarial 이미지 (정규화 범위 유지됨)
    """

    images = images.clone().detach().to(device)
    targets = targets.float().to(device).view(-1, 1)

    # tanh space 초기화 (정규화 범위: [-1, 1] 기준)
    w = torch.arctanh(images * 0.999).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(iters):
        adv_images = torch.tanh(w)  # box constraint (-1~1)

        logits = model(adv_images)
        probs = torch.sigmoid(logits)

        # perturbation 크기 최소화 (L2 distance)
        loss1 = nn.MSELoss()(adv_images, images)

        # 확률을 반대 클래스 쪽으로 보내도록 유도
        # 예: target=1이면 → 예측 확률 ↓
        #     target=0이면 → 예측 확률 ↑
        f_loss = torch.abs(probs - (1.0 - targets))
        loss2 = torch.mean(f_loss)

        # 최종 손실
        loss = loss1 + c * loss2
        
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.tanh(w).detach()  # [-1, 1] 정규화된 적대적 이미지 반환
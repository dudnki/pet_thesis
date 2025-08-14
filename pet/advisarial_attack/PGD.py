import torch
import torch.nn as nn


def pgd_attack(model, image, target, epsilon, alpha=0.01, iters=10):
    """
    Projected Gradient Descent (PGD)
    Args:
        model: PyTorch 모델
        image: 입력 이미지 텐서 [B, C, H, W]
        target: 정답 라벨 [B, 1]
        epsilon: 최대 perturbation 크기
        alpha: step size
        iters: 반복 횟수
    Returns:
        적대적 이미지
    """
    perturbed = image.clone().detach().requires_grad_(True)
    target = target.float()
    
    for _ in range(iters):
        output = model(perturbed)
        loss = nn.BCEWithLogitsLoss()(output, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv = perturbed + alpha * perturbed.grad.sign()
            eta = torch.clamp(adv - image, min=-epsilon, max=epsilon)
            perturbed = torch.clamp(image + eta, -1, 1).detach_().requires_grad_(True)

    return perturbed.detach()
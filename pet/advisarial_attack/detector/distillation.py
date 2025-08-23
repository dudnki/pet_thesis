from pet.advisarial_attack.detector.detector import ResNetDiscriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from pet.advisarial_attack.detector.PatchZeroDetector import PatchZeroDetector
from pet.advisarial_attack.detector.consistency_detector import ConsistencyDetector

def train_detector_with_distillation(teacher_model, student_model, dataset, detec_name, device='cuda', epochs=10, lr=1e-3, T=4.0, alpha=0.3):
    """
    Teacher 모델의 지식을 증류하여 Student 모델을 학습시키고,
    검증 손실이 가장 낮았던 모델과 테스트 로더를 반환합니다.
    """
    # 데이터 분할 및 로더 설정
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(1337))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16)
    
    # Student 모델 및 옵티마이저 설정
    student_model.to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    
    # 손실 함수 정의
    loss_ce = nn.BCEWithLogitsLoss()  # Hard Label을 위한 손실
    loss_kd = nn.KLDivLoss(reduction='batchmean')  # Soft Label을 위한 손실
    
    best_val_loss = float('inf')
    best_model_state = None

    print(f"\n--- Training Student '{detec_name}' with Distillation ---")
    for epoch in range(epochs):
        student_model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device).float().reshape(-1, 1)

            # Teacher의 예측 (Soft Label, 그래디언트 계산 X)
            with torch.no_grad():
                teacher_logits = teacher_model(xb)

            # Student의 예측
            student_logits = student_model(xb)

            # 1. CE Loss (Hard Label): Student가 실제 정답을 얼마나 잘 맞추는지 계산
            ce = loss_ce(student_logits, yb)

            # 2. KD Loss (Soft Label): Student와 Teacher의 예측이 얼마나 비슷한지 계산
            # 온도를 적용하여 부드러운 확률 분포 생성
            student_soft_out = F.log_softmax(student_logits / T, dim=1)
            teacher_soft_out = F.softmax(teacher_logits / T, dim=1)
            
            # KL Divergence로 두 분포의 차이를 계산 (T^2를 곱해 스케일 보정)
            kd = loss_kd(student_soft_out, teacher_soft_out) * (T * T)

            # 3. 최종 손실: 두 손실을 alpha 가중치로 조합
            total_loss = alpha * ce + (1 - alpha) * kd

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Validation: 검증 시에는 실제 정답(Hard Label)에 대한 성능만 평가
        student_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device).float().reshape(-1, 1)
                val_logits = student_model(xb)
                total_val_loss += loss_ce(val_logits, yb).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = student_model.state_dict().copy()

    print(f"Training finished for '{detec_name}'. Loading best model (val_loss: {best_val_loss:.4f}).")
    if best_model_state:
        student_model.load_state_dict(best_model_state)
    
    return student_model, test_loader

# (ConsistencyDetector, PatchZeroDetector, ResNetDiscriminator 클래스 정의가 있다고 가정)

def get_detectors_for_experiment(victim_model):
    """
    실험에 사용할 탐지기 모델들을 역할에 맞게 반환합니다.
    - Teacher: 지식을 전달할 고성능 ResNet-50 모델
    - Standard Students: 일반 방식으로 학습할 모델들
    - Distillation Student: 지식 증류로 학습할 저사양 ResNet-18 모델
    """
    
    # 1. Teacher 모델 정의 (지식 증류에 사용될 전문가 모델)
    teacher_model = ResNetDiscriminator(base_model='resnet50')
    
    # 2. 일반 방식으로 학습할 Student 모델들
    standard_students = {
        
        'consistency': ConsistencyDetector(
            victim_model=victim_model,
            blur_sigmas=(0.8, 1.2), bit_depths=(3, 4, 5),
            down_up_scales=(0.75, 0.9), gauss_noise_stds=(0.01, 0.02),
            temp=1.0, hidden=16, drop=0.2
        ),
        'patchzero': PatchZeroDetector(
            victim_model=victim_model,
            win=64, stride=32, pad=16, topk=1, expand=8, fill="zero"
        )
    }
    
    # 3. 지식 증류로 학습할 Student 모델
    distillation_student = {
        'resnet18_student': ResNetDiscriminator(base_model='resnet18')
    }
    
    return teacher_model, standard_students, distillation_student
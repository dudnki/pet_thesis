import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # 신규 평가 함수에서 사용되므로 미리 import
import torchvision.models as models
import torchvision.transforms as transforms
from fuc_torch.values import metrics 
from FGSM import fgsm_attack
from PGD import pgd_attack
from CW import cw_attack
import json
from torch.utils.data import random_split

from fuc_torch.model_torch import get_transfer_model
from fuc_torch.data_load import EyeDataset, EyeDataset_test

from PatchZeroDetector import PatchZeroDetector
from consistency_detector import ConsistencyDetector



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 1337




def sigmoid_np(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def plot_global_channel_distributions(clean_tensor, adv_tensor, title_prefix=""):
    """
    전체 배치의 clean/adv 이미지에 대해 채널별 분포를 시각화.
    clean_tensor, adv_tensor: [B, 3, H, W] 형태의 텐서
    """
    clean_np = clean_tensor.cpu().detach().numpy()
    adv_np = adv_tensor.cpu().detach().numpy()
    channels = ['Red', 'Green', 'Blue']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title_prefix} - Global Channel Distributions (Clean vs Adv)', fontsize=16)
    for i in range(3):
        clean_values = clean_np[:, i, :, :].flatten()
        adv_values = adv_np[:, i, :, :].flatten()
        axs[i].hist(clean_values, bins=100, alpha=0.5, label='Clean', color='blue', density=True)
        axs[i].hist(adv_values, bins=100, alpha=0.5, label='Adversarial', color='red', density=True)
        axs[i].set_title(f'{channels[i]} Channel')
        axs[i].legend()
        axs[i].set_xlim([-1, 1])
        # 평균 및 표준편차 출력
        c_mean, c_std = clean_values.mean(), clean_values.std()
        a_mean, a_std = adv_values.mean(), adv_values.std()
        print(f"[{channels[i]}] Clean   - mean: {c_mean:.4f}, std: {c_std:.4f}")
        print(f"[{channels[i]}] Adv     - mean: {a_mean:.4f}, std: {a_std:.4f}")
        print("-" * 40)
    plt.tight_layout()
    # 저장 경로를 확인하고 필요시 폴더를 생성하세요.
    # os.makedirs('pet/advisarial_attack/attacked_anal/', exist_ok=True)
    # plt.savefig(f'pet/advisarial_attack/attacked_anal/'+title_prefix+'.png')
    plt.show() # 로컬 환경에서 바로 확인하기 위해 show()로 변경하거나, 기존 코드를 유지하세요.


class ResNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        # 기존 fc 레이어 제거
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # [B, 512, 1, 1]
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),  # ResNet50의 fc 입력은 2048
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 512]
        x = self.classifier(x)         # [B, 1]
        return x  # logits


def create_attack_dataset(model, dataloader, attack_fn, attack_name, epsilon, device=DEVICE):
    """
    원본(clean) 데이터와 적대적(adversarial) 데이터를 생성하여
    하나의 통합된 TensorDataset으로 반환합니다.

    Args:
        model (nn.Module): 공격 대상이 되는 원본 모델 (Victim model).
        dataloader (DataLoader): 원본 데이터 로더.
        attack_fn (function): 사용할 공격 함수 (e.g., fgsm_attack).
        attack_name (str): 공격 이름 (e.g., 'fgsm').
        epsilon (float): 공격 강도.
        device (str): 연산에 사용할 장치.

    Returns:
        TensorDataset: (이미지, 판별기 레이블, 원본 레이블)을 포함하는 데이터셋.
        torch.Tensor: 시각화를 위한 모든 원본 이미지 텐서.
        torch.Tensor: 시각화를 위한 모든 적대적 예제 텐서.
    """
    model.eval()
    model.to(device)

    clean_images, adv_images = [], []
    clean_detector_labels, adv_detector_labels = [], []
    clean_original_labels, adv_original_labels = [], []

    print("Step 1/2: Preparing clean data...")
    for data, target in tqdm(dataloader, desc="[DATA PREP] Clean"):
        clean_images.append(data.cpu())
        clean_detector_labels.append(torch.zeros(data.size(0)))
        clean_original_labels.append(target.cpu())

    print(f"Step 2/2: Generating adversarial data ({attack_name.upper()}, eps={epsilon})...")
    for data, target in tqdm(dataloader, desc=f"[DATA PREP] {attack_name.upper()}"):
        data_cuda, target_cuda = data.to(device), target.to(device).reshape(-1, 1).float()

        if attack_name == 'fgsm':
            data_cuda.requires_grad = True
            output = model(data_cuda)
            loss = nn.BCEWithLogitsLoss()(output, target_cuda)
            model.zero_grad()
            loss.backward()
            data_grad = data_cuda.grad.detach()
            adv_data = attack_fn(data_cuda.detach(), epsilon, data_grad)
        elif attack_name == 'cw':
            c = epsilon * 300
            adv_data = attack_fn(model, data_cuda, target_cuda, c=c, iters=500, lr=0.01, device=device)
        else: # PGD
            adv_data = attack_fn(model, data_cuda, target_cuda, epsilon)
        
        adv_images.append(adv_data.cpu().detach())
        adv_detector_labels.append(torch.ones(adv_data.size(0)))
        adv_original_labels.append(target.cpu())
            
    # 모든 데이터를 하나의 텐서로 통합
    all_clean_images = torch.cat(clean_images, dim=0)
    all_adv_images = torch.cat(adv_images, dim=0)
    
    X_all = torch.cat([all_clean_images, all_adv_images], dim=0)
    y_detector_all = torch.cat(clean_detector_labels + adv_detector_labels, dim=0)
    y_original_all = torch.cat(clean_original_labels + adv_original_labels, dim=0)

    dataset = TensorDataset(X_all, y_detector_all, y_original_all)
    
    return dataset, all_clean_images, all_adv_images


def train_detector(detector, dataset, detec_name, device=DEVICE, epochs=10, lr=1e-3):
    """
    주어진 데이터셋으로 판별 모델(detector)을 학습시키고, 
    가장 성능이 좋았던 모델과 테스트 데이터 로더를 반환합니다.

    Args:
        detector (nn.Module): 학습시킬 판별 모델.
        dataset (TensorDataset): 학습에 사용할 전체 데이터셋.
        detec_name (str): 판별 모델의 이름 (로그 출력용).
        device (str): 연산에 사용할 장치.
        epochs (int): 총 학습 에포크.
        lr (float): 학습률.

    Returns:
        nn.Module: 검증 데이터셋에서 가장 손실이 낮았던 상태의 학습된 판별 모델.
        DataLoader: 평가에 사용할 테스트 데이터 로더.
    """
    train_set, val_set, test_set = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    detector.to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // 2)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n--- Training detector '{detec_name}' ---")
    for epoch in tqdm(range(epochs), desc=f"Training {detec_name}"):
        detector.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device).float().reshape(-1, 1)
            optimizer.zero_grad()
            out = detector(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        detector.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device).float().reshape(-1, 1)
                total_val_loss += criterion(detector(xb), yb).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = detector.state_dict().copy()
            # print(f"INFO: New best model found at epoch {epoch+1}")

    print(f"Training finished for '{detec_name}'. Loading best model (val_loss: {best_val_loss:.4f}).")
    if best_model_state:
        detector.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state was saved. Using model from the last epoch.")
    
    return detector, test_loader





def evaluate_defense_pipeline(victim_model, detector, test_loader, device='cuda'):
    """판별 모델과 예측 모델로 구성된 전체 파이프라인의 성능을 종합적으로 평가합니다."""
    victim_model.eval().to(device)
    detector.eval().to(device)

    all_detector_true = []
    all_detector_pred = []

    A_count = 0  # 정상 데이터가 정상으로 통과 후, 예측도 성공한 수
    B_count = 0  # 공격 데이터가 공격으로 탐지되어 차단된 수
    total_clean = 0
    total_adv = 0

    with torch.no_grad():
        # test_loader는 (이미지, 판별용 레이블, 원본 레이블)을 반환
        for xb, y_detector, y_original in test_loader:
            xb, y_detector, y_original = xb.to(device), y_detector.to(device), y_original.to(device)
            
            total_clean += (y_detector == 0).sum().item()
            total_adv += (y_detector == 1).sum().item()

            # 1. 판별 모델(Detector)의 결정
            detector_logits = detector(xb)
            detector_preds = (torch.sigmoid(detector_logits) > 0.5).int().squeeze()

            all_detector_true.extend(y_detector.cpu().numpy())
            all_detector_pred.extend(detector_preds.cpu().numpy())

            # 2. 판별 모델의 결정을 바탕으로 A, B 카운트
            # B 계산: 실제 공격(y_detector==1)인데, 공격이라고 탐지(detector_preds==1)된 경우
            B_count += ((y_detector == 1) & (detector_preds == 1)).sum().item()
            
            # A 계산을 위해 '정상'으로 통과된 데이터만 필터링
            normal_mask = (detector_preds == 0)
            if normal_mask.sum() > 0:
                passed_data = xb[normal_mask]
                passed_original_labels = y_original[normal_mask]
                passed_detector_labels = y_detector[normal_mask]

                # 예측 모델(Victim Model) 추론
                victim_logits = victim_model(passed_data)
                victim_preds = (torch.sigmoid(victim_logits) > 0.5).int().squeeze()
                
                # A 계산: 통과된 데이터 중 실제 정상(passed_detector_labels==0)이고, 예측도 정답(victim_preds == passed_original_labels)인 경우
                # squeeze(0) -> 배치 사이즈 1일때 차원 에러 방지
                A_count += ((passed_detector_labels == 0) & (victim_preds.squeeze(0) == passed_original_labels.squeeze(0))).sum().item()

    # --- 최종 결과 리포트 ---
    print("\n" + "="*50)
    print("           Defense Pipeline Evaluation Results")
    print("="*50)

    # 1. 판별 모델 자체 성능 리포트
    cm = confusion_matrix(all_detector_true, all_detector_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[1] Detector Performance")
    print(f"  - True Negatives (TN): {tn:4d} (Clean -> Clean)")
    print(f"  - False Positives (FP): {fp:4d} (Clean -> Attack) - 멀쩡한 데이터 차단")
    print(f"  - False Negatives (FN): {fn:4d} (Attack -> Clean) - 공격 데이터 통과 (위험!)")
    print(f"  - True Positives (TP): {tp:4d} (Attack -> Attack) - 공격 탐지 성공")
    print(f"  - Detector Accuracy: {(tn+tp)/(tn+fp+fn+tp):.4f}")

    # 2. 최종 파이프라인 성능 리포트
    final_accuracy = (A_count + B_count) / (total_clean + total_adv)
    clean_original_acc = A_count / total_clean if total_clean > 0 else 0
    attack_defense_rate = B_count / total_adv if total_adv > 0 else 0
    
    print(f"\n[2] End-to-End System Performance")
    print(f"  - A (Clean & Correct): {A_count:4d} / {total_clean:4d} (정상 데이터 중 최종 정답 수)")
    print(f"  - B (Attack & Blocked): {B_count:4d} / {total_adv:4d} (공격 데이터 중 방어 성공 수)")
    print(f"\n  - Performance on Clean Data (A / Total Clean): {clean_original_acc:.4f}")
    print(f"  - Defense Rate on Attacked Data (B / Total Attack): {attack_defense_rate:.4f}")
    print(f"  - >> Final System Accuracy ((A+B)/Total): {final_accuracy:.4f} <<")
    print("="*50)
    
    # JSON 파일 저장을 위해 결과 딕셔너리 반환
    accuracy = (tn + tp) / (tn + fp + fn + tp)

    results = {
        # 정수 값들은 int()로 변환
        'detector_TN': int(tn),
        'detector_FP': int(fp),
        'detector_FN': int(fn),
        'detector_TP': int(tp),
        'final_A_count': int(A_count),
        'final_B_count': int(B_count),
        'total_clean': int(total_clean),
        'total_adv': int(total_adv),
        
        # 실수(소수점) 값들은 float()로 변환
        'detector_accuracy': float(accuracy),
        'final_system_accuracy': float(final_accuracy)
    }
    return results



attack_method_dict = {
    'fgsm': fgsm_attack,
    'cw': cw_attack,
    'pgd': pgd_attack,
}

path = 'dog/eye/ultrasound'
# 데이터 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


num_classes = 1
model_dict = {
    #'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    #'vgg16': get_transfer_model('vgg16', num_classes),
    #'resnet50': get_transfer_model('resnet50', num_classes),
    #'densenet121': get_transfer_model('densenet121', num_classes),
    #'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    #'vit_base': get_transfer_model('vit_base', num_classes),
    #'custom_efficient': get_transfer_model('custom_efficient', num_classes),
    'efficientnet_v2_1': get_transfer_model('efficientnet_v2_1', num_classes),
}


def get_fresh_detectors(victim_model):
    return {
        'consistency': ConsistencyDetector(
            victim_model=victim_model,
            blur_sigmas=(0.8, 1.2), bit_depths=(3, 4, 5),
            down_up_scales=(0.75, 0.9), gauss_noise_stds=(0.01, 0.02),
            temp=1.0, hidden=16, drop=0.2
        ),
        'patchzero': PatchZeroDetector(
            victim_model=victim_model,
            win=64, stride=32, pad=16, topk=1, expand=8, fill="zero"
        ),
        'resnet': ResNetDiscriminator()
    }



dataset = EyeDataset_test(path, disease='cataract', transform=transform)
print('data_len:', len(dataset))
model_path = './pet/model_save'
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for model_name, model in model_dict.items():
    # 1) 모델 로드
    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # 2) 전체 결과 저장을 위한 딕셔너리
    overall_results = {}

    # 3) attack_method → epsilon 순회 (데이터 생성)
    for attack_name, attack_fn in attack_method_dict.items():
        eps_results = {}
        for eps in [0.005, 0.01, 0.02]:
            
            #  Step 1: 데이터셋을 공격 조건별로 **한 번만** 생성합니다.
            print("\n" + "="*60)
            print(f"Creating dataset for ATTACK: {attack_name.upper()}, EPSILON: {eps}")
            print("="*60)
            adversarial_dataset, clean_tensors, adv_tensors = create_attack_dataset(
                model, loader, attack_fn, attack_name, eps
            )

            # (옵션) 데이터 분포 시각화도 여기서 한 번만 수행
            # title = f"{model_name}_{attack_name.upper()}_eps={eps}"
            # plot_global_channel_distributions(clean_tensors, adv_tensors, title_prefix=title)

            detec_results = {}
            
            #  Step 2: 모든 탐지기에 대해 생성된 데이터셋을 **재사용**하여 학습합니다.
            
            # 매번 새로운 학습을 위해 초기화된 탐지기 모델들을 가져옵니다.
            detectors_to_train = get_fresh_detectors(model)
            
            for detec_name, detector_instance in detectors_to_train.items():
                
                # 2-1) 판별 모델 훈련
                trained_detector, test_loader = train_detector(
                    detector=detector_instance,      # 훈련시킬 새로운 탐지기 모델
                    dataset=adversarial_dataset, # 재사용하는 데이터셋
                    detec_name=detec_name
                )
                
                # 2-2) 전체 파이프라인 성능 평가
                pipeline_metrics = evaluate_defense_pipeline(
                    model, trained_detector, test_loader
                )
                
                detec_results[detec_name] = pipeline_metrics

            # 결과 저장
            key_eps = eps * 300 if attack_name == 'cw' else eps
            eps_results[key_eps] = detec_results
        
        overall_results[attack_name] = eps_results

    # 4) 모델별 최종 결과 저장
    output_filename = f"pet/advisarial_attack/adversarial_discriminator_eval/adversarial_discriminator_{model_name}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=4)
    print(f"\n All experiments for {model_name} are complete. Results saved to {output_filename}")

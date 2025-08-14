import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from fuc_torch.values import *
from torch.utils.data import DataLoader
from FGSM import fgsm_attack
from PGD import pgd_attack
from CW import cw_attack
from PatchZeroDetector import PatchZeroDetector
from consistency_detector import ConsistencyDetector
from fuc_torch.model_torch import get_transfer_model
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from fuc_torch.data_load import EyeDataset, EyeDataset_test
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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
    plt.savefig(f'pet/advisarial_attack/attacked_anal/'+title_prefix+'.png')
    


# 안전한 시그모이드 함수 (overflow 방지)
def sigmoid_np(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


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

    
def adversarial_discriminator(model, detector, dataloader, attack_fn, attack_name, epsilon, model_name, threshold=0.5):
    model.eval()
    model.to('cuda')

    all_targets = []
    clean_images = []
    adv_images = []

    # 1. clean 데이터 (정상 예측 결과)
    for data, target in tqdm(dataloader, desc=f"[CLEAN BASELINE]"):
        data = data.to('cuda')
        clean_images.append(data.detach().cpu())
        all_targets.append(torch.zeros(data.size(0)))  # label 0

    # 2. 공격된 데이터
    for data, target in tqdm(dataloader, desc=f"[{attack_name}] ε={epsilon}"):
        data, target = data.to('cuda'), target.to('cuda').reshape(-1, 1)

        if attack_name == 'fgsm':
            data.requires_grad = True
            output = model(data)
            loss = nn.BCEWithLogitsLoss()(output, target)
            model.zero_grad()
            loss.backward()
            
            data_grad = data.grad.detach()                           
            data = data.detach()                                     
            data.requires_grad_(False) 

            adv_data = attack_fn(data, epsilon, data_grad)
        elif attack_name == 'cw':
            c = epsilon * 300
            adv_data = attack_fn(model, data, target, c=c, iters=500, lr=0.01, device='cuda')
        else:
            adv_data = attack_fn(model, data, target, epsilon)

        adv_images.append(adv_data.detach().cpu())
        all_targets.append(torch.ones(adv_data.size(0)))  # label 1
            

    # 3. 입력 생성 (clean + adv)
    clean_tensor = torch.cat(clean_images, dim=0)  # [N, 3, 224, 224]
    adv_tensor = torch.cat(adv_images, dim=0)      # [N, 3, 224, 224]
    X_all = torch.cat(clean_images + adv_images, dim=0).detach()  # [2B, 3, 224, 224]
    y_all = torch.cat(all_targets, dim=0).detach()                # [2B]

    if attack_name == 'cw':
        plot_global_channel_distributions(clean_tensor, adv_tensor, title_prefix=f"{attack_name.upper()}_C={c}")
    else:
        plot_global_channel_distributions(clean_tensor, adv_tensor, title_prefix=f"{attack_name.upper()}_epsilon={epsilon}")
    detector_dataset = TensorDataset(X_all, y_all)
    
    train_set, val_set, test_set = torch.utils.data.random_split(detector_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(1337))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=0)
    
    
    detector.to('cuda')
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) 
    epochs = 10
    
    for epoch in tqdm(range(epochs)):
        detector.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to('cuda'), yb.to('cuda').reshape(-1, 1)
            optimizer.zero_grad()
            out = detector(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()


        detector.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                    xb, yb = xb.to('cuda'), yb.to('cuda').float().reshape(-1, 1)
                    out = detector(xb)
                    loss = criterion(out, yb)
                    total_loss += loss.item()
                    probs = torch.sigmoid(out).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_labels.extend(yb.cpu().numpy().flatten())
        avg_loss = total_loss / len(val_loader)
        print(f"{detec_name}_Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
        print(f'{detec_name}_val_loss:', avg_loss)
    
    detector.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to('cuda')
            logits = detector(xb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()  
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().flatten())
    
    print(metrics(all_labels, all_preds, all_probs))
    return metrics(all_labels, all_preds, all_probs)
    
    
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
    'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    #'vit_base': get_transfer_model('vit_base', num_classes),
    #'custom_efficient': get_transfer_model('custom_efficient', num_classes),
}



dataset = EyeDataset_test(path, disease='cataract', transform=transform)
print('data_len:', len(dataset))
model_path = './pet/nested_model_save'
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for model_name, model in model_dict.items():
    # 1) 모델 로드
    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)

    # 2) detectors 정의 
    detectors = {
        'consistency': ConsistencyDetector(
        victim_model=model,
        blur_sigmas=(0.8, 1.2),
        bit_depths=(3,4,5),
        down_up_scales=(0.75, 0.9),
        gauss_noise_stds=(0.01, 0.02),
        temp=1.0,
        hidden=16, drop=0.2
    ),
        'patchzero': PatchZeroDetector(
            victim_model=model,
            win=64, stride=32,
            pad=16, topk=1, expand=8,
            fill="zero"
        ),
        'resnet': ResNetDiscriminator()
    }

    # 3) 결과 저장용 딕셔너리
    detect_results = {}

    # 4) attack_method → epsilon → detector 순회
    for attack_name, attack_fn in attack_method_dict.items():
        eps_results = {}
        for eps in [0.005, 0.01, 0.02]:
            detec_results = {}
            for detec_name, detector in detectors.items():
                if attack_name == 'cw':
                    key_eps = eps * 300
                else:
                    key_eps = eps
                detec_results[detec_name] = adversarial_discriminator(
                    model, detector, loader,
                    attack_fn, attack_name, eps, model_name,
                    threshold=0.3
                )
            eps_results[key_eps] = detec_results
        detect_results[attack_name] = eps_results

    # 5) 모델별 결과 저장
    with open(f"pet/advisarial_attack/adversarial_discriminator_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(detect_results, f, ensure_ascii=False, indent=4)



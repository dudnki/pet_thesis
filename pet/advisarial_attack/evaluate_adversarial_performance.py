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
from fuc_torch.model_torch import get_transfer_model
from fuc_torch.data_load import EyeDataset, EyeDataset_test

# 안전한 시그모이드 함수 (overflow 방지)
def sigmoid_np(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def normal_eval(model, dataloader, model_name, threshold=0.5):
    model.eval()
    model.to('cuda')
    
    
    all_targets = []
    clean_outputs = []
    for data, target in tqdm(dataloader):
        data, target = data.to('cuda'), target.to('cuda').reshape(-1, 1)
        
        with torch.no_grad():
            clean_output = model(data)
        all_targets.append(target.detach().cpu())
        clean_outputs.append(clean_output.detach().cpu())
    
    y_true = torch.cat(all_targets).numpy().astype(np.float32)
    clean_probs = sigmoid_np(torch.cat(clean_outputs).numpy())
    clean_preds = (clean_probs > threshold).astype(np.float32)
    
    print(f"\n===== {model_name}")
    print(" Clean 성능")
    print(metrics(y_true, clean_preds, clean_probs))
    return metrics(y_true, clean_preds, clean_probs)
    
    
def evaluate_adversarial_performance(model, dataloader, attack_fn, attack_name, epsilon, model_name, threshold=0.5):
    model.eval()
    model.to('cuda')

    all_targets = []
    adv_outputs = []

    for data, target in tqdm(dataloader, desc=f"[{attack_name}] ε={epsilon}"):
        data, target = data.to('cuda'), target.to('cuda').reshape(-1, 1)


        # FGSM은 외부 gradient 사용
        if attack_name == 'fgsm':
            data.requires_grad = True
            output = model(data)
            loss = nn.BCEWithLogitsLoss()(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            adv_data = attack_fn(data, epsilon, data_grad)
        elif attack_name == 'cw':
            c = epsilon * 300
            adv_data = attack_fn(model, data, target, c=c, iters=500, lr=0.01, device='cuda')
        else:
            adv_data = attack_fn(model, data, target, epsilon)
            
        
        with torch.no_grad():
            adv_output = model(adv_data)

        all_targets.append(target.detach().cpu())
        adv_outputs.append(adv_output.detach().cpu())

    # 결과 정리
    y_true = torch.cat(all_targets).numpy().astype(np.float32)
    adv_probs = sigmoid_np(torch.cat(adv_outputs).numpy())

    adv_preds = (adv_probs > threshold).astype(np.float32)
    
    # 출력
    if attack_name == 'cw':
        print(f"\n===== {model_name} Attack: {attack_name}, c={c} =====")
    else:
        print(f"\n===== {model_name} Attack: {attack_name}, ε={epsilon} =====")
    print(" Adversarial 성능")
    print(metrics(y_true, adv_preds, adv_probs))
    return metrics(y_true, adv_preds, adv_probs)
    
    
attack_method_dict = {
    'cw': cw_attack,
    'fgsm': fgsm_attack,
    'pgd': pgd_attack,
}

path = 'dog/eye/ultrasound'
# 데이터 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  
        std=[0.5, 0.5, 0.5]
    )
])


num_classes = 1
model_dict = {
    'custom_efficient': get_transfer_model('custom_efficient', num_classes),
    'convnext_large': get_transfer_model('convnext_large', num_classes),
    'swin_large': get_transfer_model('swin_large', num_classes),
    'nfnet_f6': get_transfer_model('nfnet_f6', num_classes),
    'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    'efficientnet_v2_1': get_transfer_model('efficientnet_v2_1', num_classes),
    'efficientnet_v2': get_transfer_model('efficientnet_v2', num_classes),
    'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    'vgg16': get_transfer_model('vgg16', num_classes),
    'resnet50': get_transfer_model('resnet50', num_classes),
    'densenet121': get_transfer_model('densenet121', num_classes),
    'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    'vit_base': get_transfer_model('vit_base', num_classes),
}


dataset = EyeDataset_test(path, disease='cataract', transform=transform)
print('data_len:', len(dataset))
model_path = './pet/model_save'
loader = DataLoader(dataset, batch_size=32, shuffle=True)
file_path = "pet/advisarial_attack/adversl_eval.json"

# 기존 데이터 불러오기
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            eval_dict = json.load(f)
        except json.JSONDecodeError:
            eval_dict = {}
else:
    eval_dict = {}

# 새 결과 추가
for model_name, model in model_dict.items():
    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)

    attack_dict = {}
    for attack_name, attack_fn in attack_method_dict.items():
        eps_dict = {}
        for eps in [0.005, 0.01, 0.02]:
            if attack_name == 'cw':
                eps_dict[eps * 300] = evaluate_adversarial_performance(
                    model, loader, attack_fn, attack_name, eps, model_name, threshold=0.3
                )
            else:
                eps_dict[eps] = evaluate_adversarial_performance(
                    model, loader, attack_fn, attack_name, eps, model_name, threshold=0.3
                )
        attack_dict[attack_name] = eps_dict

    attack_dict['plane'] = normal_eval(model, loader, model_name, threshold=0.3)
    eval_dict[model_name] = attack_dict  # 기존 동일 model_name이면 덮어씀

    # 저장
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, ensure_ascii=False, indent=4)

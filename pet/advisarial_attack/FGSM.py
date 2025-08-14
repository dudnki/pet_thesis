import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
from fuc_torch.values import *
from fuc_torch.model_torch import get_transfer_model
from fuc_torch.data_load import EyeDataset, EyeDataset_test
from tqdm import tqdm

def sigmoid(x):
    return 1/(1+np.exp(-x))

def fgsm_attack(image, epsilon, data_grad):
    """
    Fast Gradient Sign Method (FGSM)
    Args:
        image (Tensor): 입력 이미지 [B, C, H, W]
        epsilon (float): perturbation 범위
        data_grad (Tensor): gradient of loss w.r.t input
    Returns:
        perturbed image
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -1, 1)
'''
def adversarial_attack(model, test_loader, epsilon, adversarial_method):
    model.to('cuda')
    model.eval()
    pl_all_probs = []
    adv_all_probs = []
    all_targets = []

    for data, target in tqdm(test_loader):
        data, target = data.to('cuda'), target.to('cuda').reshape(-1, 1)
        data.requires_grad = True
        # 순전파
        output = model(data)
        loss = nn.BCEWithLogitsLoss()(output, target)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            pl_output = model(data)
        
        pl_all_probs.append(pl_output.to('cpu'))

        
        # gradient 가져오기
        data_grad = data.grad.data

        # FGSM 공격 수행
        perturbed_data = adversarial_method(data, epsilon, data_grad)
        perturbed_data = perturbed_data.detach()

        # 적대적 샘플로 예측
        with torch.no_grad():
            adv_output = model(perturbed_data)

        
        adv_all_probs.append(adv_output.to('cpu'))
        all_targets.append(target.to('cpu'))
    

    pl_all_probs = torch.cat(pl_all_probs).numpy()
    adv_all_probs = torch.cat(adv_all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()


    adv_all_preds = (sigmoid(adv_all_probs) > 0.3).astype(np.float32)
    pl_all_preds = (sigmoid(pl_all_probs) > 0.3).astype(np.float32)  
        


    print('normal')
    print(metrics(all_targets, pl_all_preds, pl_all_probs))
    print('adversal')
    print(metrics(all_targets, adv_all_preds, adv_all_probs))
        


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

loader = DataLoader(dataset, batch_size=32, shuffle=True)

attack_method = [fgsm_attack]
model_path = './pet/nested_model_save'
epsilons = [0.1, 0.4, 0.7]
for model_name, model in model_dict.items():

    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    for attack in attack_method:
        for eps in epsilons:
            adversarial_attack(model, loader, eps, attack)
            '''
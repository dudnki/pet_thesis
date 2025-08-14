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



def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # clamp to [0,1]
    return torch.clamp(perturbed_image, 0, 1)

def imshow(img, title):
        img = img.squeeze().detach().cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0), cmap='gray')
        plt.title(title)
        plt.axis('off')


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


def adversarial_attack(model, loader, attack_method, epsilon=0.03):
    
    image, label = next(iter(loader))
    image, label = image.to('cuda'), label.to('cuda').reshape(-1, 1)
    model = model.to('cuda')
    
    
    model.eval()
    
    image.requires_grad = True
    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]
    print(init_pred.item() == label.item())
    if init_pred.item() == label.item():
        loss = nn.BCEWithLogitsLoss()(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data

        perturbed_image = attack_method(image, epsilon, data_grad)
        output_adv = model(perturbed_image)
        final_pred = output_adv.max(1, keepdim=True)[1]
        
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        imshow(image, f"Original (Label: {label.item()})")
        plt.subplot(1,2,2)
        imshow(perturbed_image, f"Adversarial (Pred: {final_pred.item()})")
        plt.show()
        
        
        
dataset = EyeDataset_test(path, disease='cataract', transform=transform)
print('data_len:', len(dataset))

loader = DataLoader(dataset, batch_size=1, shuffle=True)

attack_method = [fgsm_attack]
model_path = './pet/nested_model_save'
for model_name, model in model_dict.items():

    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    for attack in attack_method:
        adversarial_attack(model, loader, attack)
    
    


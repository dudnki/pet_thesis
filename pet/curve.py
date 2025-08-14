import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fuc_torch.data_load import EyeDataset
from torch.utils.data import DataLoader
from fuc_torch.values import plot_multiple_roc_plot, plot_multiple_pr_curves
from fuc_torch.model_torch import get_transfer_model


path = 'dog/eye/ultrasound'
# 데이터 준비
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  
        std=[0.5, 0.5, 0.5]
    )
])

num_classes = 1
model_dict = {
    'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    'vgg16': get_transfer_model('vgg16', num_classes),
    'resnet50': get_transfer_model('resnet50', num_classes),
    'densenet121': get_transfer_model('densenet121', num_classes),
    'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    'vit_base': get_transfer_model('vit_base', num_classes),
}

model_names = []
trained_models = []
model_path = './pet/nested_model_save'
for model_name, model in model_dict.items():
    model_names.append(model_name)

    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")

    # state_dict 로드 및 적용
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)

    trained_models.append(model)

dataset = EyeDataset(path, disease='cataract', transform=train_transform)
print('data_len:', len(dataset))
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(1337))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=32, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, num_workers=0)




plot_multiple_roc_plot(trained_models, model_names, test_loader, device="cpu")
plot_multiple_pr_curves(trained_models, model_names, test_loader, device="cpu")




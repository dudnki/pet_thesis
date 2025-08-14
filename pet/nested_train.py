# PyTorch 기반 전체 학습 + 평가 + 설명 시각화 파이프라인

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from fuc_torch.data_load import EyeDataset
from fuc_torch.values import *
from fuc_torch.model_torch import get_transfer_model
from sklearn.model_selection import StratifiedKFold
import copy



path = 'dog/eye/ultrasound'
# 데이터 준비
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  
        std=[0.5, 0.5, 0.5]
    )
])



param_grid = [
    #{'lr': 1e-3, 'batch_size': 32, 'weight_decay': 1e-4, 'epochs': 10},   # 안정적인 기본 조합
    #{'lr': 5e-4, 'batch_size': 48, 'weight_decay': 1e-5, 'epochs': 15},   # 더 작은 lr, 더 큰 batch
    {'lr': 1e-4, 'batch_size': 32, 'weight_decay': 1e-5, 'epochs': 30},   # 느리지만 정밀하게
    #{'lr': 1e-2, 'batch_size': 16, 'weight_decay': 0,    'epochs': 10},   # 공격적인 설정 (빠른 학습 시도)
]

dataset = EyeDataset(path, disease='cataract', transform=train_transform)
print('data_len:', len(dataset))
targets = [label for _, label in dataset]
targets = torch.stack(targets).long().tolist()
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_scores = []



def train_and_evaluate(model, model_name):
    print(f'model{model_name}')
    for fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(targets)), targets)):
        print(f"\n[Fold {fold+1}]")
        
        # 내부 CV용 데이터 분할
        train_val_subset = Subset(dataset, train_val_idx)
        train_val_targets = [targets[i] for i in train_val_idx]

        best_score = -1
        best_params = None
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for params in param_grid:
            lr = params['lr']
            batch_size = params['batch_size']
            weight_decay = params['weight_decay']
            epochs = params['epochs']
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(np.zeros(len(train_val_targets)), train_val_targets):
                inner_train = Subset(train_val_subset, inner_train_idx)
                inner_val = Subset(train_val_subset, inner_val_idx)

                model = get_transfer_model(model_name, num_classes)
                model = model.cuda() if torch.cuda.is_available() else model
                
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
                train_loader = DataLoader(inner_train, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(inner_val, batch_size=batch_size)
                for epoch in tqdm(range(epochs)):
                    model.train()
                    running_loss = 0
                    for imgs, labels in train_loader:
                        imgs, labels = imgs.to('cuda'), labels.to('cuda').reshape(-1, 1)
                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        
                    scheduler.step()
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to('cuda'), labels.to('cuda').reshape(-1, 1)
                        outputs = model(imgs)
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                acc = correct / total
                inner_scores.append(acc)
                
            avg_score = np.mean(inner_scores)
            print(f"{model_name} Params {params} → Inner CV Acc: {avg_score:.4f}") # 여기 avg_score 손보기!!!!
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                
                
        print(f"{model_name} Best Params: {best_params}, Best Score:{best_score}")
        
        test_subset = Subset(dataset, test_idx)
        test_loader = DataLoader(test_subset, batch_size=best_params['batch_size'])

        train_subset = Subset(dataset, train_val_idx)
        train_loader = DataLoader(train_subset, batch_size=best_params['batch_size'], shuffle=True)

        # 모델 재학습
        model = get_transfer_model(model_name, num_classes)
        model = model.cuda() if torch.cuda.is_available() else model
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        criterion =  nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        for epoch in tqdm(range(best_params['epochs'])):
            model.train()
            running_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to('cuda'), labels.to('cuda').reshape(-1, 1)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            scheduler.step()
            print(f"{model_name} Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
        # 테스트 정확도 측정
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to('cuda'), labels.to('cuda').reshape(-1, 1)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        outer_scores.append(acc)
        print(f"{model_name} [Fold {fold+1}] Test Accuracy: {acc:.4f}")
        
    torch.save(model.state_dict() , f'./pet/nested_model_save/{model_name}.pt')
    #  최종 결과
    print(f"\n {model_name} Nested CV 평균 정확도: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
                




num_classes = 1
model_dict = {
    'nfnet_f6': get_transfer_model('nfnet_f6', num_classes),
    'convnext_large': get_transfer_model('convnext_large', num_classes),
    'swin_large': get_transfer_model('swin_large', num_classes),
    #'vgg16': get_transfer_model('vgg16', num_classes),
    #'resnet50': get_transfer_model('resnet50', num_classes),
    #'densenet121': get_transfer_model('densenet121', num_classes),
    #'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    #'vit_base': get_transfer_model('vit_base', num_classes),
    #'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    #'efficientnet_v2': get_transfer_model('efficientnet_v2', num_classes),
    #'custom_efficient': get_transfer_model('custom_efficient', num_classes),
    
}

for model_name, model in model_dict.items():
    train_and_evaluate(model, model_name)

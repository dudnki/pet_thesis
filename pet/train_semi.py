# PyTorch 기반 전체 학습 + 평가 + 설명 시각화 파이프라인

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fuc_torch.data_load import EyeDataset
from fuc_torch.values import *
from fuc_torch.model_torch import get_transfer_model


def mixup(x1, y1, x2, y2, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix





def get_last_conv_layer(model, model_name):
    if model_name == 'vgg16':
        return model.features[-1]  # MaxPool2d
    elif model_name == 'resnet50':
        return model.layer4[-1]  # 마지막 Bottleneck 블록
    elif model_name == 'densenet121':
        return model.features[-1]  # 마지막 ReLU
    elif model_name == 'efficientnet_b0':
        return model.features[-1]  # Conv2d(1280, ...)
    elif model_name == 'convnext_tiny':
        return model.features[-1]  # LayerNorm2d
    elif model_name == 'swin_tiny':
        return model.backbone.layers[-1].blocks[-1].norm1


def extract_features_and_labels(dataloader):
    all_x = []
    all_y = []
    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x).numpy()
    all_y = torch.cat(all_y).numpy()
    return all_x, all_y

def mixmatch(model, x_l, y_l, x_u, K=2, T=0.5, alpha=0.75):
  
    model.eval()
    with torch.no_grad():
        preds = [torch.sigmoid(model(x_u)) for _ in range(K)]
        p_avg = torch.stack(preds).mean(0)
        # binary classification일 경우 sharpening은 아래처럼
        p_avg_cat = torch.cat([1 - p_avg, p_avg], dim=1)
        p_sharp = sharpen(p_avg_cat, T=T)[:, 1:2]  # binary 확률만 추출
        y_u = p_sharp.detach()

    x_all = torch.cat([x_l, x_u], dim=0)
    y_all = torch.cat([y_l, y_u], dim=0)

    idx = torch.randperm(x_all.size(0))
    x2, y2 = x_all[idx], y_all[idx]

    x_mix, y_mix = mixup(x_all, y_all, x2, y2, alpha=alpha)

    return x_mix, y_mix

path = 'dog/eye/ultrasound'
# 데이터 준비
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  
        std=[0.5, 0.5, 0.5]
    )
])

dataset = EyeDataset(path, disease='cataract', transform=train_transform)
print('data_len:', len(dataset))
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(1337))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=64, num_workers=0)
test_loader = DataLoader(test_set, batch_size=64, num_workers=0)

# 전체 데이터 추출 (예: train_loader 기준)
x_train, y_train = extract_features_and_labels(train_loader)

# t-SNE 시각화
tsne(x_train, y_train)

# UMAP 시각화
umap_vis(x_train, y_train)



# 학습 및 평가

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader):
    print(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 스케줄러 (CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)  
    epochs = 30
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).reshape(-1, 1)
            optimizer.zero_grad()
            out = model(xb)
            loss_orig = criterion(out, yb)
            
            # mix
            idx = torch.randperm(xb.size(0))
            xb2, yb2 = xb[idx], yb[idx]
            x_mix, y_mix = mixup(xb, yb, xb2, yb2, alpha=0.75)
            logits_mix = model(x_mix)
            loss_mix = criterion(logits_mix, y_mix)
            
            
            loss = 0.5 * loss_orig + 0.5 * loss_mix
            
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            
        scheduler.step()
        
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float().reshape(-1, 1)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item()
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy().flatten())
        avg_loss = total_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
        print('val_loss:', avg_loss)
        

    # 평가
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()  
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().flatten())

    y_true = all_labels
    y_pred = all_preds
    y_score = all_probs

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_score)
    print(f"[{model_name}] Accuracy: {acc:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, AUC: {auc_val:.3f}")

    print(classification_report(all_labels, all_preds))
    save_met(model_name, recall, precision, f1, auc_val)
    #save_csv(model_name, round(acc,3), 0, 0, 0)  # loss/odds ratio는 별도 계산 시 반영
    plot_confusion_matrix(all_labels, all_preds, model_name)
    roc_plot(y_true, y_score, model_name)

    # 시각화 설명
    for image, true in test_loader:
        if true == 1:
            x_test = image
            break
        else: continue
    sample_img = x_test[0].unsqueeze(0).to(device)
    #gradcam = make_gradcam_heatmap(sample_img, model, last_conv_layer)
    #save_htm(sample_img.cpu(), gradcam[0], model_name)

    saliency = make_sal(sample_img, model)
    save_sal(sample_img.cpu(), saliency[0], model_name)

    lime_explain(sample_img.cpu(), model, model_name, save_dir="./lime_results")
    shap_explain(sample_img.cpu(), model, model_name, test_set, save_dir="./shap_results")
    
    #모델 저장
    
    torch.save(model.state_dict() , f'./pet/model_save/{model_name}.pt')

num_classes = 1
model_dict = {
    'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'efficientnet_v2_1': get_transfer_model('efficientnet_v2_1', num_classes),
    'efficientnet_v2': get_transfer_model('efficientnet_v2', num_classes),
    'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    'vgg16': get_transfer_model('vgg16', num_classes),
    'resnet50': get_transfer_model('resnet50', num_classes),
    'densenet121': get_transfer_model('densenet121', num_classes),
    'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    'vit_base': get_transfer_model('vit_base', num_classes),
}

for model_name, model in model_dict.items():
    print(f"Training {model_name}")
    last_conv_layer = get_last_conv_layer(model, model_name)
    train_and_evaluate(model, model_name, train_loader, val_loader, test_loader)
    break

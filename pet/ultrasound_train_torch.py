# PyTorch 기반 전체 학습 + 평가 + 설명 시각화 파이프라인
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

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
from fuc_torch.data_load import EyeDataset, EyeDataset_test
from fuc_torch.values import *
from fuc_torch.model_torch import get_transfer_model
from fuc_torch.early_stop import EarlyStopper



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

dataset = EyeDataset(path, disease='cataract', transform=train_transform)
test_set = EyeDataset_test(path, disease='cataract', transform=train_transform)
print('data_len:', len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1337))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=32, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=32, num_workers=0, pin_memory=True)



# 전체 데이터 추출 (예: train_loader 기준)
#x_train, y_train = extract_features_and_labels(train_loader)

# t-SNE 시각화
#tsne(x_train, y_train)

# UMAP 시각화
#umap_vis(x_train, y_train)

# 학습 및 평가

# --- 변경점: AMP/GradScaler + grad checkpointing + accumulation + channels_last + no_grad/autocast + 메모리 정리 ---

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader):

    print(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # timm 백본이면 gradient checkpointing 지원하는 경우가 많음
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)

    # CNN 메모리 효율 (가능하면 channels_last 사용)
    try:
        model = model.to(memory_format=torch.channels_last)
        use_channels_last = True
    except Exception:
        use_channels_last = False

    epochs = 25
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler()  # AMP
    early_stopper = EarlyStopper(patience=3)

    ACCUM = 4          # gradient accumulation 스텝 (유효 배치 = batch_size * ACCUM)
    optimizer.zero_grad(set_to_none=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        for step, (xb, yb) in enumerate(train_loader):
            if use_channels_last:
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float().reshape(-1, 1)

            with torch.cuda.amp.autocast():
                out = model(xb)
                loss = criterion(out, yb) / ACCUM   # 누산을 위해 나눔

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUM  # 표시용: 원래 스케일로 환원

        scheduler.step()

        # ---------- Validation ----------
        model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for xb, yb in val_loader:
                if use_channels_last:
                    xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float().reshape(-1, 1)

                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item()

                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                

        avg_loss = total_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / max(1, len(train_loader)):.4f}  -  ValLoss: {avg_loss:.4f}  -  ValAcc: {val_acc:.4f}")
        if early_stopper(avg_loss, model):
            break 
        
        
        # 에폭 사이에 메모리 청소
        torch.cuda.empty_cache()
    early_stopper.load_best_model(model)
    
        
        
        
    # ---------- Test ----------
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for xb, yb in test_loader:
            if use_channels_last:
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.numpy().flatten().tolist())

    y_true = all_labels
    y_pred = all_preds
    y_score = all_probs

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_score)
    print(f"[{model_name}] Accuracy: {acc:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, AUC: {auc_val:.3f}")

    print(classification_report(y_true, y_pred))
    save_met(model_name, recall, precision, f1, auc_val)
    plot_confusion_matrix(y_true, y_pred, model_name)
    roc_plot(y_true, y_score, model_name)

    # ---------- 설명(설명 단계는 GPU 메모리 아끼려고 CPU로) ----------
    # 샘플 하나만 뽑아서 설명
    x_test = None
    for image, true in test_loader:
        if true[0] == 1:
            x_test = image
            break
    if x_test is None:
        x_test = next(iter(test_loader))[0]  # fallback

    sample_img = x_test[0].unsqueeze(0).cpu()  # 설명은 CPU로 이동
    # Grad-CAM은 Transformer에 바로 쓰기 어려워서 conv 모델에서만 사용 권장
    # gradcam = make_gradcam_heatmap(sample_img.to(device), model, last_conv_layer)
    # save_htm(sample_img, gradcam[0], model_name)

    saliency = make_sal(sample_img, model.cpu())   # 모델도 CPU로 잠시 이동해 계산
    save_sal(sample_img, saliency[0], model_name)

    lime_explain(sample_img, model.cpu(), model_name, save_dir="./lime_results")
    shap_explain(sample_img, model.cpu(), model_name, test_set, save_dir="./shap_results")

    # 모델 저장 (원하면 다시 GPU로 보내지 말고 그대로 저장)
    model_path = f'./pet/model_save/{model_name}.pt'
    torch.save(model.state_dict(), model_path)

    # 큰 객체 정리
    del sample_img, saliency
    torch.cuda.empty_cache()

num_classes = 1
model_dict = {
    'custom_efficient': get_transfer_model('custom_efficient', num_classes),
    #'convnext_large': get_transfer_model('convnext_large', num_classes),
    #'swin_large': get_transfer_model('swin_large', num_classes),
    #'nfnet_f6': get_transfer_model('nfnet_f6', num_classes),
    #'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'efficientnet_v2_1': get_transfer_model('efficientnet_v2_1', num_classes),
    #'efficientnet_v2': get_transfer_model('efficientnet_v2', num_classes),
    #'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    #'vgg16': get_transfer_model('vgg16', num_classes),
    #'resnet50': get_transfer_model('resnet50', num_classes),
    #'densenet121': get_transfer_model('densenet121', num_classes),
    #'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    #'vit_base': get_transfer_model('vit_base', num_classes),
}

for model_name, model in model_dict.items():
    print(f"Training {model_name}")
    last_conv_layer = get_last_conv_layer(model, model_name)
    train_and_evaluate(model, model_name, train_loader, val_loader, test_loader)


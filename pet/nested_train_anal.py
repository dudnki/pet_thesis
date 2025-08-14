import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fuc_torch.data_load import EyeDataset, EyeDataset_test
from torch.utils.data import DataLoader
from fuc_torch.values import *
from fuc_torch.model_torch import get_transfer_model


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
    'nfnet_f6': get_transfer_model('nfnet_f6', num_classes),
    'convnext_large': get_transfer_model('convnext_large', num_classes),
    'swin_large': get_transfer_model('swin_large', num_classes),
    #'swin_tiny': get_transfer_model('swin_tiny', num_classes),
    #'vgg16': get_transfer_model('vgg16', num_classes),
    #'resnet50': get_transfer_model('resnet50', num_classes),
    #'densenet121': get_transfer_model('densenet121', num_classes),
    #'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes),
    #'convnext_tiny': get_transfer_model('convnext_tiny', num_classes),
    #'vit_base': get_transfer_model('vit_base', num_classes),
    #'custom_efficient': get_transfer_model('custom_efficient', num_classes),
}

    
dataset = EyeDataset_test(path, disease='cataract', transform=transform)
print('data_len:', len(dataset))

test_loader = DataLoader(dataset, batch_size=32, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, model_name, test_loader):
    model.eval()
    model.to(device)
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
        if true[0] == 1:
            x_test = image
            break
        else: continue
    sample_img = x_test[0].unsqueeze(0).to(device)
    #gradcam = make_gradcam_heatmap(sample_img, model, last_conv_layer)
    #save_htm(sample_img.cpu(), gradcam[0], model_name)

    saliency = make_sal(sample_img, model)
    save_sal(sample_img.cpu(), saliency[0], model_name, save_dir='./nested_sal')

    lime_explain(sample_img.cpu(), model, model_name, save_dir="./nested_lime_results")
    shap_explain(sample_img.cpu(), model, model_name, dataset, save_dir="./nested_shap_results")




model_path = './pet/nested_model_save'
for model_name, model in model_dict.items():

    model_file = os.path.join(model_path, f'{model_name}.pt')
    print(f"Loading model from: {model_file}")

    # state_dict 로드 및 적용
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    evaluate(model, model_name, test_loader)

    

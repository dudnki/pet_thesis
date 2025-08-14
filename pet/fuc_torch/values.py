

import shap
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve, roc_auc_score, average_precision_score

from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import label_binarize
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from PIL import Image
from skimage.segmentation import mark_boundaries
from lime import lime_image

# 저장 디렉토리 생성 함수
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 이미지 전처리
img_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def preprocess(folder_path):
    class_folders = ['0', '1']
    selected_class = random.choice(class_folders)
    class_folder_path = os.path.join(folder_path, selected_class)
    image_files = os.listdir(class_folder_path)
    selected_image_name = random.choice(image_files)
    img_path = os.path.join(class_folder_path, selected_image_name)

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return img_tensor, img_path

# Grad-CAM

def make_gradcam_heatmap(x, model, target_layer):
    model.eval()
    heatmaps = []

    for img in x:
        img = img.unsqueeze(0).requires_grad_()
        activations = {}

        def forward_hook(module, input, output):
            activations['value'] = output.detach()

        handle = target_layer.register_forward_hook(forward_hook)
        output = model(img)
        pred_class = output.argmax(dim=1)
        score = output[0, pred_class]

        model.zero_grad()
        score.backward(retain_graph=True)
        gradients = img.grad

        pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
        act = activations['value'].squeeze()

        for i in range(act.shape[0]):
            act[i, :, :] *= pooled_grads[i]

        heatmap = act.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        heatmaps.append(heatmap)

        handle.remove()

    return heatmaps

def save_htm(img_tensor, heatmap, model_name, cam_path="cam.jpg", alpha=0.4):
    save_dir = os.path.join("./results", model_name, "htm")
    ensure_dir(save_dir)

    img = img_tensor.squeeze().permute(1, 2, 0).numpy()
    img = np.uint8(255 * img)

    heatmap = np.uint8(255 * heatmap)
    jet_colors = plt.cm.jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(heatmap)
    axs[1].set_title('Heatmap')
    axs[1].axis('off')

    axs[2].imshow(superimposed_img)
    axs[2].set_title('Superimposed Image')
    axs[2].axis('off')

    fig.savefig(os.path.join(save_dir, f"{cam_path}"))
    plt.close()

# Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred, model_name):
    class_names = ['0', '1']
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    save_dir = os.path.join("./results", model_name, "cm")
    ensure_dir(save_dir)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"cm_{model_name}.png"))
    plt.close()

# ROC Curve
def roc_plot(y_true, y_score, model_name):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_true_bin = label_binarize(y_true, classes=[0, 1])

    
    fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_score)
    roc_auc = auc(fpr, tpr)

    save_dir = os.path.join("./results/roc")
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f"roc_{model_name}.png"))
    plt.close()

# Precision-Recall Curve
def pr_plot(y_true, y_score, model_name):
    y_true = np.array(y_true).reshape(1, -1).squeeze()
    y_score = np.array(y_score)
    y_true_bin = label_binarize(y_true, classes=[0, 1])

    precision, recall, _ = precision_recall_curve(y_true_bin[:, 1], y_score[:, 1])
    pr_auc = auc(recall, precision)

    save_dir = os.path.join("./results", model_name, "pr")
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_dir, f"pr_{model_name}.png"))
    plt.close()

# Classification Report 저장
def save_classification_report(y_true, y_pred, model_name):
    class_names = ['0', '1']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    save_dir = os.path.join("./results", model_name)
    ensure_dir(save_dir)

    df_report.to_csv(os.path.join(save_dir, f"classification_report_{model_name}.csv"))

# Metric 저장
def save_metrics(y_true, y_pred, y_score, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(label_binarize(y_true, classes=[0, 1]), y_score, average='macro')

    df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "Recall", "Precision", "AUC"],
        "Value": [acc, f1, recall, precision, auc_score]
    })

    save_dir = os.path.join("./results", model_name)
    ensure_dir(save_dir)

    df.to_csv(os.path.join(save_dir, f"metrics_{model_name}.csv"), index=False)

def metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(label_binarize(y_true, classes=[0, 1]), y_score, average='macro')
    metric = { "Accuracy" : acc, "F1_Score":f1, "Recall":recall, "Precision": precision, "AUC":auc_score}
    return metric

def make_sal(x, model):
    model.eval()
    sal_list = []

    for img in x:
        img = img.unsqueeze(0).requires_grad_()

        output = model(img)
        score = output.max(1)[0]  # max logit value
        model.zero_grad()
        score.backward()

        saliency = img.grad.data.abs().squeeze().max(0)[0]  # max over channels
        saliency = saliency.cpu().numpy()
        saliency = 255 * (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        saliency = saliency.astype(np.uint8)
        sal_list.append(saliency)

    return sal_list

def lime_explain(img_tensor, model, model_name, save_dir="./results"):
    model.eval()
    device = next(model.parameters()).device  # 모델이 위치한 장치 가져오기

    # LIME에 넣기 위한 numpy 이미지
    img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
    img_np = np.uint8(255 * img_np)

    # 모델 예측 함수 (numpy → torch → softmax → numpy)
    def batch_predict(images):
        model.eval()
        batch = torch.stack(
            [transforms.ToTensor()(Image.fromarray(img)) for img in images], dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
        return logits.sigmoid().detach().cpu().numpy()

    # LIME explainer 설정
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # 시각화 및 저장
    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # 저장 경로 설정
    lime_dir = os.path.join(save_dir, model_name, "lime")
    os.makedirs(lime_dir, exist_ok=True)
    out_path = os.path.join(lime_dir, f"lime_{model_name}.png")

    # 시각화 저장
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.savefig(out_path)
    plt.close()
    




def shap_explain(img_tensor, model, model_name, dataset, save_dir="./shap_results"):
    import shap
    import torch
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device

    #  배경 데이터 생성
    def get_background_data(dataset, num_samples=30):
        background = []
        count = 0
        for img, label in dataset:
            if label == 0:
                background.append(img)  # img: [3, 256, 256]

                count += 1
            if count >= num_samples:
                break
        return torch.stack(background).to(device)  # [30, 3, 256, 256]

    background_tensor = get_background_data(dataset)  


    #  설명 대상 이미지
    img_tensor = img_tensor.to(device)  # [1, 3, 256, 256]
  

    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(img_tensor)
    shap_val = shap_values[..., 0][0]
    shap_image = shap_val.transpose(1, 2, 0)

    #  시각화용 numpy 변환
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.uint8(255 * img_np)
    

    #  저장
    shap_dir = os.path.join(save_dir, model_name, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    shap.image_plot(shap_image, img_np, show=False)
    plt.savefig(os.path.join(shap_dir, f"shap_{model_name}.png"))
    plt.close()



def save_csv(model_name, test_acc, test_loss, csv_filename='result.csv'):
    header = ['model_name', 'accuracy', 'loss']
    new_row = [model_name, test_acc, test_loss]

    # 파일이 없으면 생성 + 헤더 포함 저장
    if not os.path.exists(csv_filename):
        df = pd.DataFrame([new_row], columns=header)
        df.to_csv(csv_filename, index=False)
    else:
        df = pd.read_csv(csv_filename)

        # 모델명 존재 시 업데이트, 없으면 추가
        if model_name in df['model_name'].values:
            df.loc[df['model_name'] == model_name, ['accuracy', 'loss']] = new_row[1:]
        else:
            df = pd.concat([df, pd.DataFrame([new_row], columns=header)], ignore_index=True)

        df.to_csv(csv_filename, index=False)

def save_sal(img_tensor, saliency_map, model_name, save_dir="./sal"):
    os.makedirs(save_dir, exist_ok=True)

    # 텐서 형식 → NumPy 변환 및 채널 순서 변환
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
        if img.shape[0] == 3:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))
    else:
        img = img_tensor
        
        
    img = np.uint8(255 * img)
    
    
    img = img.squeeze(0)             # (3, 256, 256)
    img = img.transpose(1, 2, 0)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(saliency_map, cmap='jet')
    plt.axis('off')
    plt.title('Saliency map')

    plt.subplot(1, 3, 3)
    plt.imshow(img, alpha=0.5)
    plt.imshow(saliency_map, cmap='jet', alpha=0.9)
    plt.axis('off')
    plt.title('Superimposed Image')

    plt.savefig(os.path.join(save_dir, f"{model_name}.png"))
    plt.close()


def save_met(model_name, recall, precision, f1score, auc, csv_filename='./results/metrics.csv'):
    header = ['model_name', 'recall', 'precision', 'f1score', 'auc']
    new_row = [model_name, recall, precision, f1score, auc]
    csv_directory = './results'
    os.makedirs(csv_directory, exist_ok=True)

    # CSV 파일이 존재하지 않을 경우 파일 생성 및 헤더 추가
    if not os.path.exists(csv_filename):
        df = pd.DataFrame([new_row], columns=header)
        df.to_csv(csv_filename, index=False, header=False)
    else:
        # CSV 파일 읽기 (헤더 없음으로 읽기)
        df = pd.read_csv(csv_filename, header=None, names=header)

        # model_name에 해당하는 행을 찾기
        if model_name in df['model_name'].values:
            df.loc[df['model_name'] == model_name, ['recall', 'precision', 'f1score', 'auc']] = new_row[1:]
        else:
            # model_name이 존재하지 않을 경우 새로운 행 추가
            df = pd.concat([df, pd.DataFrame([new_row], columns=header)], ignore_index=True)

        # 수정된 데이터프레임을 CSV 파일에 저장
        df.columns = ['model_name', 'recall', 'precision', 'f1score', 'auc']
        df.to_csv(csv_filename, index=False, header=False)


def tsne(x, y):
    x=x.reshape(x.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(x)
    
    save_dir = os.path.join("./results/tsne")
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap='viridis')
    plt.legend()
    plt.xlim(-80, 60)
    plt.ylim(-80, 60)
    plt.colorbar()
    plt.title('t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(save_dir, f"image_tsne.png"))
    

def umap_vis(data, labels):
    plt.clf()
    data_flattened = data.reshape(data.shape[0], -1)
    umap_model = umap.UMAP(n_components=2)
    
    save_dir = os.path.join("./results/umap")
    ensure_dir(save_dir)

    embedding = umap_model.fit_transform(data_flattened)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
    plt.legend()
    plt.xlim(0, 14)
    plt.ylim(0, 14)
    plt.colorbar()
    plt.title('UMAP')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(os.path.join(save_dir, f"image_umap.png"))
    
    
    
def plot_multiple_pr_curves(models, model_names, test_loader, device="cpu"):

    plt.figure(figsize=(8, 6))
    save_dir = os.path.join("./results/pr")
    ensure_dir(save_dir)
    
    all_X, all_y = [], []
    for X_batch, y_batch in test_loader:
        all_X.append(X_batch)
        all_y.append(y_batch)
    X_test = torch.cat(all_X).to(device)
    y_test = torch.cat(all_y).cpu().numpy()

    for model, name in zip(models, model_names):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)  # shape: [N, 1]
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 확률값으로 변환

        # PR 곡선 및 AP 계산
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap_score = average_precision_score(y_test, probs)

        # 그래프에 추가
        plt.plot(recall, precision, label=f"{name} (AP={ap_score:.2f})")

    # 그래프 꾸미기
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_all_model.png"))
    plt.close()
'''
def plot_multiple_roc_plot(models, model_names, test_loader):
    plt.figure(figsize=(8, 6))
    save_dir = os.path.join("./results/roc")
    ensure_dir(save_dir)
    for model, name in zip(models, model_names):
        for X_test, y_test in test_loader:
            # 예측 확률 또는 결정 함수 값 얻기
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]  # 양성 클래스 확률
    
            y_true_bin = label_binarize(y_test, classes=[0, 1])

            
            fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AP={roc_auc:.2f})')


    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f"roc_all_model.png"))
    plt.close()
'''  
def plot_multiple_roc_plot(models, model_names, test_loader, device="cpu"):
    plt.figure(figsize=(8, 6))
    save_dir = "./results/roc"
    ensure_dir(save_dir)

    # 전체 테스트셋 수집
    all_X, all_y = [], []
    for X_batch, y_batch in test_loader:
        all_X.append(X_batch)
        all_y.append(y_batch)
    X_test = torch.cat(all_X).to(device)
    y_test = torch.cat(all_y).cpu().numpy()

    for model, name in zip(models, model_names):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)  # shape: [N, 1]
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 확률값으로 변환

        # ROC 계산
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        # 그래프에 추가
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    # 기준선, 스타일 설정
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Models)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_all_models.png"))
    plt.close()
    
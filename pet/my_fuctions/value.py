from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tensorflow as tf
import keras
from IPython.display import Image, display
import matplotlib as mpl
from sklearn.metrics import *
import pandas as pd
import random
from PIL import Image
from sklearn.manifold import TSNE
import umap as umap
from sklearn.preprocessing import label_binarize
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, linregress
from lime.lime_tabular import LimeTabularExplainer
import lime
from lime import lime_image
import matplotlib.cm as cm
from skimage.segmentation import mark_boundaries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def data_split(x_data, y_data):
    splits = np.array_split(np.arange(len(x_data)), 10)
    
    x_splits = [x_data[split] for split in splits]
    y_splits = [y_data[split] for split in splits]
    
    return x_splits, y_splits

def model_split(model):
    models = [model for _ in range(10)]
    return models

#### ----------------------------수정
def roc(test_result, true_labels, fname, num, title):
    pred_labels = np.argmax(test_result, axis=1)
    y_true = np.argmax(true_labels, axis=1)
    
    save_dir = os.path.join("./results", "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    colors = ['red', 'green']
    disease = ['0', '1']
    plt.figure(figsize=(10, 8))

    for i, color in enumerate(colors):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, pred_labels == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        label = '{} (area = {:.3f})'.format(disease[i], roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)
        
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    file_path = os.path.join(save_dir, f"roc_{num}_{title}.png")
    plt.savefig(file_path)
    plt.show()

def roc3(test_result, true_labels, fname, num, title):
    pred_labels = np.argmax(test_result, axis=1)
    y_true = np.argmax(true_labels, axis=1)

    fpr = {}
    tpr = {}
    roc_auc = {}

    colors = ['green', 'blue', 'orange']
    disease= ['0', '1', '2']
    plt.figure(figsize=(10, 8))

    for i, color in enumerate(colors):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, pred_labels == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        label = '{} (area = {:.3f})'.format(disease[i], roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)

    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(r"/root/ming/{}/roc_{}_{}.png".format(fname, num, title))
    plt.show()

def pr(test_result, true_labels, fname, num, title):
    precision = dict()
    recall = dict()
    pr_auc = dict()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['red', 'green']
    disease= ['0', '1']

    for i, color in enumerate(colors) :
        true_labels_binary = label_binarize(true_labels, classes=[0, 1])[:, i]
        
        precision[i], recall[i], _ = precision_recall_curve(true_labels_binary, test_result[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        ax.plot(recall[i], precision[i], color=color, lw=2, label='{} (area = {:.3f})'.format(disease[i], pr_auc[i]))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.savefig(r"/root/ming/{}/pr_{}_{}.png".format(fname, num, title))
    plt.show()

def pr3(test_result, true_labels, fname, num, title):
    precision = dict()
    recall = dict()
    pr_auc = dict()

    colors = ['green', 'blue', 'orange']
    disease= ['0', '1', '2']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, color in enumerate(colors):
        precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], test_result[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

        ax.plot(recall[i], precision[i], color=color, lw=2, label='{} (area = {:.3f})'.format(disease[i], pr_auc[i]))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.savefig(r"/root/ming/{}/pr_{}_{}.png".format(fname, num, title))
    plt.show()

def cm_cls(model, x_test, y_test, fname, num, title):
    plt.clf()
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    class_names = ['0', '1']

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.savefig(r"/root/ming/{}/cm_{}_{}.png".format(fname, num, title))
    plt.show()

    print("\nClassification Report :")
    print(classification_report(y_true_classes, y_pred_classes, labels=range(len(class_names)), target_names=class_names))
    report_dict = classification_report(y_true_classes, y_pred_classes, labels=range(len(class_names)), target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict)
    report_df.to_csv(r"/root/ming/{}/cr_{}_{}.csv".format(fname, num, title))

img_size = (256, 256)

def specificity_score(y_true, y_pred):
    tn = 0
    fp = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i != j:
                fp += confusion_matrix(y_true[i], y_pred[i])[0, 1]
            else:
                tn += confusion_matrix(y_true[i], y_pred[i])[0, 0] 
    return tn / (tn + fp) if (tn + fp) != 0 else 0  

def cal_met(y_pred, y_preds, y_true, y_trues, fname, num, title) :

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_trues, y_preds, multi_class='ovo')
    sensitivity = recall_score(y_true, y_pred, average='macro')
    specificity = specificity_score(y_true, y_pred)
    
    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #specificity = tn / (tn + fp)
    
    route = "/root/ming/{}/".format(fname)

    output_file = os.path.join(route, "cal_{}_{}.csv".format(num, title))
    report_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "AUC Score", "Sensitivity", "Specificity"],
        "Value": [accuracy, f1, auc, sensitivity, specificity]
    })
    report_df.to_csv(output_file, index=False)

def preprocess(folder_path):
    selected_image_path = None
    class_folders = ['0', '1']
    selected_class = random.choice(class_folders)
    class_folder_path = os.path.join(folder_path, selected_class)
    image_files = os.listdir(class_folder_path)
    selected_image_name = random.choice(image_files)
    img_path = os.path.join(class_folder_path, selected_image_name)

    img = keras.utils.load_img(img_path, target_size=img_size)
    array1 = keras.utils.img_to_array(img)
    array1 = np.expand_dims(array1, axis=0)
    array = array1.transpose(0, 3, 1, 2)
    return array, img_path

def make_gradcam_heatmap(x, model, last_conv_layer_name, pred_index=None):
    htm_list = []
    for image in x:
        img = tf.expand_dims(image, axis=0)
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        htm_list.append(heatmap)
    return htm_list

def save_htm(img, heatmap, model_name, cam_path="cam.jpg", alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet_colors = cm.jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * 255  # [0, 1] 범위의 img 값을 [0, 255]로 변환
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img.astype(np.float32))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(heatmap)
    axs[1].set_title('Heatmap')
    axs[1].axis('off')

    superimposed_img = np.array(superimposed_img)
    axs[2].imshow(superimposed_img.astype(np.uint8))
    axs[2].set_title('Superimposed Image')
    axs[2].axis('off')

    plt.savefig(r"./htm/{}.png".format(model_name))

def make_sal(x, model) :
    sal_list = []
    for img in x :
      img_array = tf.keras.preprocessing.image.img_to_array(img)
      img_array = tf.expand_dims(img_array, axis=0)

      with tf.GradientTape() as tape:
          inputs = tf.cast(img_array, tf.float32)
          tape.watch(inputs)
          predictions = model(inputs)

      gradients = tape.gradient(predictions, inputs)
      grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)

      normalized_tensor = tf.cast(
          255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
          / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
          tf.uint8,
      )
      normalized_tensor = tf.squeeze(normalized_tensor)
      sal_list.append(normalized_tensor)

    return sal_list

def save_sal(img, saliency_map, model_name):
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

    plt.savefig(r"./sal/{}.png".format(model_name))

def odd_ratio(x_htm, x_sal, y):
    x_htm = np.array(x_htm)
    x_sal = np.array(x_sal)

    x_htm_flat = x_htm.reshape(x_htm.shape[0], -1)  
    x_sal_flat = x_sal.reshape(x_sal.shape[0], -1)  

    model = LogisticRegression()

    y_flat = np.argmax(y, axis=1)

    model.fit(np.hstack((x_htm_flat, x_sal_flat)), y_flat)

    coefficients = model.coef_[0]

    odds_ratio_htm = np.exp(coefficients[0])  
    odds_ratio_sal = np.exp(coefficients[1])  

    return odds_ratio_htm, odds_ratio_sal

def tsne(x, y):
    x=x.reshape(x.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('./tsne.png')
    plt.show()

def umap_vis(data, labels):
    plt.clf()
    data_flattened = data.reshape(data.shape[0], -1)
    umap_model = umap.UMAP(n_components=2)

    embedding = umap_model.fit_transform(data_flattened)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('UMAP')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('./umap.png')
    plt.show()

def histogram(x, y) :
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(x, bins=20, kde=True, color='blue', alpha=0.7)
    plt.xlabel('X')
    plt.title('Histogram of X')

    plt.subplot(1, 2, 2)
    sns.histplot(y, bins=20, kde=True, color='green', alpha=0.7)
    plt.xlabel('Y')
    plt.title('Histogram of Y')

    plt.tight_layout()
    plt.savefig("./hist.png")

def lime(img_array, model, model_name):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0].astype('double'), model.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig(f"./lime/{model_name}.png")

def report(x_test, y_test, model, model_name) :
  class_names = ['0', '1']

  y_pred = model.predict(x_test)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true_classes = np.argmax(y_test, axis=1)
  print(f"Classification Report({model_name}):")
  print(classification_report(y_true_classes, y_pred_classes, labels=range(len(class_names)), target_names=class_names))


def cca(X_train, Y_train, n_comp=1) :
    cca = CCA(n_components=n_comp)
    cca.fit(X_train.reshape(len(X_train), -1), Y_train)  # Flatten X_train for CCA

    X_c, Y_c = cca.transform(X_train.reshape(len(X_train), -1), Y_train)

    n_samples = X_train.shape[0]
    n_remove = int(0.1 * n_samples)
    idx_to_keep = np.argsort(X_c[:, 0])[n_remove:-n_remove]

    X_c_filtered = X_c[idx_to_keep]
    Y_c_filtered = Y_c[idx_to_keep]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_c_filtered[:, 0], Y_c_filtered[:, 0])
    plt.title('CCA Scatter Plot')
    plt.xlabel('CCA Component 1')
    plt.ylabel('CCA Component 2')
    plt.grid(True)

    corr, _ = pearsonr(X_c_filtered[:, 0], Y_c_filtered[:, 0])
    r_squared = corr ** 2

    # annotate를 사용하여 상관 계수 값 출력
    plt.annotate(f'Pearson Correlation: {corr:.2f}\nR-squared: {r_squared:.2f}',
                 xy=(0.5, 0.5), xycoords='axes fraction',
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig("./cca.png")

    plt.clf()

    plt.hist(X_c_filtered[:, 0], bins=30, color='blue', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./hist.png")

def save_csv(model_name, test_acc, test_loss, csv_filename='result.csv'):
    header = ['model_name', 'accuracy', 'loss']
    new_row = [model_name, test_acc, test_loss]

    # CSV 파일이 존재하지 않을 경우 파일 생성 및 헤더 추가
    if not os.path.exists(csv_filename):
        df = pd.DataFrame([new_row], columns=header)
        df.to_csv(csv_filename, index=False, header=False)
    else:
        # CSV 파일 읽기 (헤더 없음으로 읽기)
        df = pd.read_csv(csv_filename, header=None, names=header)

        # model_name에 해당하는 행을 찾기
        if model_name in df['model_name'].values:
            df.loc[df['model_name'] == model_name, ['accuracy', 'loss']] = new_row[1:]
        else:
            # model_name이 존재하지 않을 경우 새로운 행 추가
            df = df.append(pd.DataFrame([new_row], columns=header), ignore_index=True)

        # 수정된 데이터프레임을 CSV 파일에 저장
        df.to_csv(csv_filename, index=False, header=False)

def save_met(model_name, recall, precision, f1score, auc, csv_filename='metrics.csv'):
    header = ['model_name', 'recall', 'precision', 'f1score', 'auc']
    new_row = [model_name, recall, precision, f1score, auc]

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
            df = df.append(pd.DataFrame([new_row], columns=header), ignore_index=True)

        # 수정된 데이터프레임을 CSV 파일에 저장
        df.to_csv(csv_filename, index=False, header=False)

def plot_confusion_matrix(y_true, y_pred, model_name) :
    class_names = ['0', '1']

    cm = confusion_matrix(y_true, y_pred)

    # 레이블을 가진 데이터 프레임으로 변환
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 시각화 설정
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False, annot_kws={"size": 60})

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig(r"./cm/{}.png".format(model_name))

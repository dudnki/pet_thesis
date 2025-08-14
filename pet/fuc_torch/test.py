# PyTorch 기반 전체 학습 + 평가 + 설명 시각화 파이프라인
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
dataset = EyeDataset(path, disease='cataract', transform=train_transform)
print('data_len:', len(dataset))
targets = [label for _, label in dataset]
print(targets)
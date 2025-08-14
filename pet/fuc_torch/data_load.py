import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import json


class EyeDataset(Dataset):
    def __init__(self, root_dir, disease, transform=None):
        self.root_dir = root_dir
        self.disease = disease
        self.transform = transform
        self.samples = []  # (img_path, label)
        self.data_cnt = 0
        
        for first_path in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, first_path)
            if first_path != 'cataract': continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                for json_file in os.listdir(fpath):
                    if json_file.endswith('.json'):
                        json_path = os.path.join(fpath, json_file)
                        with open(json_path, 'r', encoding='utf-8') as file:
                            dic = json.load(file)
                            img_path = os.path.join(fpath, dic['label']['label_filename'])
                            if dic['label']['label_disease_nm'] == '백내장':
                                label = ((dic['label']['label_disease_lv_1'] == '유') or
                                         (dic['label']['label_disease_lv_2'] == '유') or
                                         (dic['label']['label_disease_lv_3'] == '유'))
                            else: label = 0
                            self.samples.append((img_path, label))
                            self.data_cnt +=1
                            if self.data_cnt % 2000 == 0: break
                    
                        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx ):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float32)  # float32로 변환
        return img, label
    
    
class EyeDataset_test(Dataset):
    def __init__(self, root_dir, disease, transform=None):
        self.root_dir = root_dir
        self.disease = disease
        self.transform = transform
        self.samples = []  # (img_path, label)
        
        for first_path in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, first_path)
            if first_path != 'cataract': continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                self.data_cnt = 0
                for json_file in os.listdir(fpath):
                    if json_file.endswith('.json'):
                        json_path = os.path.join(fpath, json_file)
                        with open(json_path, 'r', encoding='utf-8') as file:
                            dic = json.load(file)
                            img_path = os.path.join(fpath, dic['label']['label_filename'])
                            if dic['label']['label_disease_nm'] == '백내장':
                                label = ((dic['label']['label_disease_lv_1'] == '유') or
                                         (dic['label']['label_disease_lv_2'] == '유') or
                                         (dic['label']['label_disease_lv_3'] == '유'))
                            else: label = 0
                            self.data_cnt +=1
                            if self.data_cnt > 2000:
                                self.samples.append((img_path, label))
                                if self.data_cnt == 2500: break
                        
                        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx ):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float32)  # float32로 변환
        return img, label
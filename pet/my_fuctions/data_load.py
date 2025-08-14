import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch


#os.makedirs('checkpoints', exist_ok=True)
path = "dog/eye/ultrasound"
def make_test_data(path, diease):
    X = []
    y = []

    for first_path in os.listdir(path):
        folder_path = os.path.join(path, first_path)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            print(f'load_{diease}_{fname}', folder_path)
            for file_name in os.listdir(fpath):
                final_file_name = os.path.join(fpath, file_name)
                if final_file_name.endswith('.jpg'):
                    with Image.open(final_file_name) as img:
                        print(img.shape)
                        break
                        img = img.convert("RGB")  # 채널 통일
                        img = img.resize((256, 256))
                        image = np.array(img)
                        X.append(image)
                        if ('existence' in fname) and (diease == first_path):
                            y.append(1)
                        else:
                            y.append(0)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    return torch.tensor(X).permute(0, 3, 1, 2), torch.tensor(y)

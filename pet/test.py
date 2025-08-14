import pandas as pd
from fuc_torch.model_torch import get_transfer_model
from torchsummary import summary
import torch

num_classes = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    
    'efficientnet_b0': get_transfer_model('efficientnet_b0', num_classes).to(device),
}
for model_name, model in model_dict.items():
    print(f"Training {model_name}")
    summary(model, input_size=(3, 224, 224))
    print(model)
    break
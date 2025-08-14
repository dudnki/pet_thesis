from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinTinyClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.backbone.reset_classifier(0)  # ❗ 반드시 head 제거!
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, 768, 7, 7]
        x = x.permute(0, 3, 1, 2) 
        #print("after forward_features:", x.shape) 
        x = self.pool(x)                       # [B, 768, 1, 1]
        #print("pool:", x.shape)
        x = x.flatten(1)  # [B, 768]
        #print("flatten:", x.shape)
        x = self.classifier(x)                 # [B, num_classes]
        return x
    
class GenericTimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 1,
                 in_chans: int = 3, pretrained: bool = True,
                 dropout: float = 0.3, hidden: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        f = self.backbone.forward_features(x)   # (B,C,H,W) or (B,N,C) or (B,C)
        if f.dim() == 4:
            f = self.pool2d(f).flatten(1)
        elif f.dim() == 3:
            f = f.mean(dim=1)
        elif f.dim() == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected feature shape: {tuple(f.shape)}")
        return self.classifier(f)
    
    
def get_transfer_model(name, num_classes, in_chans: int = 3, pretrained: bool = True,
                       dropout: float = 0.3, hidden: int = 512):
    """
    in_chans: 1(흑백)도 지원. timm 백본은 이 인자를 그대로 전달.
    pretrained/dropout/hidden: GenericTimmClassifier에서 사용.
    """
    # --- 대형 SOTA timm 백본들 (tiny 제외) ---
    
        
    if name == 'nfnet_f6':
        return GenericTimmClassifier('dm_nfnet_f6', num_classes=num_classes, in_chans=in_chans)


    elif name == 'convnext_large':
        return GenericTimmClassifier('convnext_large', num_classes=num_classes,
                                     in_chans=in_chans, pretrained=pretrained,
                                     dropout=dropout, hidden=hidden)

    elif name == 'swin_large':
        # timm 모델명: swin_large_patch4_window7_224
        return GenericTimmClassifier('swin_large_patch4_window7_224', num_classes=num_classes,
                                     in_chans=in_chans, pretrained=pretrained,
                                     dropout=dropout, hidden=hidden)

    elif name == 'vit_large_16':
        return GenericTimmClassifier('vit_large_patch16_224', num_classes=num_classes,
                                     in_chans=in_chans, pretrained=pretrained,
                                     dropout=dropout, hidden=hidden)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),  # ResNet50의 fc 입력은 2048
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(1024, 1024),  # DenseNet121의 classifier 입력은 1024
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 1024),  # EfficientNet B0의 classifier[1] 입력은 1280
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    elif name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Sequential(
            nn.Linear(768, 512),  # ConvNeXt Tiny의 classifier[2] 입력은 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    elif name == 'swin_tiny':
        model = SwinTinyClassifier(num_classes)
        
    elif name == 'efficientnet_v2':
        model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 512),  # EfficientNet B0의 classifier[1] 입력은 1280
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif name == 'efficientnet_v2_1':
        model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 512),  # EfficientNet B0의 classifier[1] 입력은 1280
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    
    elif name == 'custom_efficient':
        model = CustomEfficientNet()
        
    
        
    else:
        raise ValueError(f"Model '{name}' is not supported.")
    
    return model


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1=32, out_3x3=32, out_5x5=32, out_pool=32):
        super(InceptionModule, self).__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3 branch (1x1 -> 3x3)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5 branch (1x1 -> 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # Pooling branch (3x3 pool -> 1x1 conv)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # 채널 방향으로 concat
    
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)

        # 원래 블록 구성 분리
        self.stem = base_model.features[:5]            # (0~4)
        self.inception = nn.Sequential(
        InceptionModule(in_channels=80),
        nn.Conv2d(128, 80, kernel_size=1),  # 다시 80채널로 축소
        nn.BatchNorm2d(80),
        nn.ReLU(inplace=True)
        )
        self.remaining = base_model.features[5:]       # (5~8)
        self.pool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[1] = nn.Sequential(
            nn.Linear(1280, 1024),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        x = self.remaining(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x    
import torch.nn as nn
import torchvision.models as models

class ResNetDiscriminator(nn.Module):
    def __init__(self, base_model='resnet50', num_classes=1):
        """
        base_model 인자에 따라 ResNet 구조를 동적으로 선택합니다.
        
        Args:
            base_model (str): 'resnet50' 또는 'resnet18' 등 torchvision에서 지원하는 모델 이름.
            num_classes (int): 최종 출력 클래스 수.
        """
        super().__init__()

        # base_model 인자에 따라 모델과 특성(feature) 수를 동적으로 결정
        if base_model == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = model.fc.in_features  # ResNet50의 fc 입력은 2048
        elif base_model == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features  # ResNet18의 fc 입력은 512
        else:
            raise ValueError(f"Unsupported base_model: {base_model}")

        # 기존 fc 레이어 제거
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
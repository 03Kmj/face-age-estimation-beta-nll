import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BetaNLL_AgePredictor(nn.Module):
    """
    ResNet18 백본을 사용하여 얼굴 이미지에서 Beta 분포의 
    두 매개변수 (Alpha, Beta)를 예측하는 모델입니다.
    """
    def __init__(self, use_pretrained=True, freeze_backbone=False, epsilon=1e-6):
        super(BetaNLL_AgePredictor, self).__init__()
        
        # 1. ResNet18 백본 로드 (사전 학습된 가중치 사용)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        
        # 2. 백본 Freeze 설정 (선택 사항: 초기 학습 안정성 확보 목적)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. 기존의 최종 FC 레이어 제거
        num_ftrs = self.backbone.fc.in_features # ResNet18의 마지막 출력 특징 벡터 크기 (512)
        self.backbone.fc = nn.Identity() # FC 레이어를 제거하고 특징 벡터만 추출하도록 설정

        # 4. Alpha, Beta 예측 헤드 연결 (2개)
        # 특징 벡터 크기: 512 -> 출력 크기: 1
        self.fc_alpha = nn.Linear(num_ftrs, 1)
        self.fc_beta = nn.Linear(num_ftrs, 1)
        
        self.epsilon = epsilon # 수치 안정성을 위한 작은 값
        
    def forward(self, x):
        # 특징 추출
        features = self.backbone(x)
        
        # 5. Softplus 활성화 함수 적용 (출력을 양수로 보장)
        # alpha = Softplus(FC_alpha(features)) + epsilon
        # beta = Softplus(FC_beta(features)) + epsilon
        alpha = F.softplus(self.fc_alpha(features)) + self.epsilon
        beta = F.softplus(self.fc_beta(features)) + self.epsilon
        
        return alpha.squeeze(), beta.squeeze()
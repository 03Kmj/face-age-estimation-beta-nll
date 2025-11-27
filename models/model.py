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
        
        # ResNet18 백본 로드 (사전 학습된 가중치 사용)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        
        # 백본 Freeze 설정
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 기존의 최종 FC 레이어 제거 및 새로운 헤드 연결 준비
        num_ftrs = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity() 

        # Alpha, Beta 예측 헤드 정의
        self.fc_alpha = nn.Linear(num_ftrs, 1)
        self.fc_beta = nn.Linear(num_ftrs, 1)
        
        self.epsilon = epsilon 
        
    def forward(self, x):
        # 특징 추출
        features = self.backbone(x)
        
        # Softplus 활성화 함수 적용 (출력을 양수로 보장)
        alpha = F.softplus(self.fc_alpha(features)) + self.epsilon
        beta = F.softplus(self.fc_beta(features)) + self.epsilon
        
        # 손실 함수로 전달하기 위해 불필요한 차원 제거 (.squeeze())
        return alpha.squeeze(), beta.squeeze()
    
    
class DeepfakeUncertaintyModel(nn.Module):
    """
    딥페이크(real/fake) 분류 + Beta 기반 불확실성 추정을 동시에 하는 모델.
    
    - cls_head: real/fake 분류 (logit 출력)
    - alpha_head, beta_head: Beta 분포 파라미터 (불확실성)
    """
    def __init__(self, use_pretrained=True, freeze_backbone=False, epsilon=1e-6):
        super(DeepfakeUncertaintyModel, self).__init__()

        # ResNet18 백본 공유
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 분류용 Head (real/fake)
        self.fc_cls   = nn.Linear(num_ftrs, 1)

        # 불확실성(Beta 분포) Head
        self.fc_alpha = nn.Linear(num_ftrs, 1)
        self.fc_beta  = nn.Linear(num_ftrs, 1)

        self.epsilon = epsilon

    def forward(self, x):
        # 공통 특징 추출
        features = self.backbone(x)          # (B, num_ftrs)

        # 1) real/fake 분류용 logit
        logit = self.fc_cls(features).squeeze(-1)   # (B,)

        # 2) Beta 분포 파라미터
        alpha = F.softplus(self.fc_alpha(features)) + self.epsilon   # (B, 1) -> 양수
        beta  = F.softplus(self.fc_beta(features))  + self.epsilon   # (B, 1)

        alpha = alpha.squeeze(-1)   # (B,)
        beta  = beta.squeeze(-1)    # (B,)

        # 최종 출력: (분류 logit, alpha, beta)
        return logit, alpha, beta
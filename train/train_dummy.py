# train/train_dummy.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import torch
import torch.nn as nn
import torch.optim as optim

# 파일 위치 변경에 따른 import 경로 수정 (models 폴더에서 가져오기)
from models.model import BetaNLL_AgePredictor 
from models.losses import Beta_NLL_Loss

# --- 환경 설정 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon GPU (MPS)를 사용합니다.")
else:
    device = torch.device("cpu")
    print("⚠️ CPU를 사용합니다.")

# --- 하이퍼파라미터 ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
BETA_REWEIGHT = 0.5 

# --- 모델 및 손실 함수 인스턴스화 ---
model = BetaNLL_AgePredictor(use_pretrained=True, freeze_backbone=True).to(device)
criterion = Beta_NLL_Loss(beta_reweight=BETA_REWEIGHT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 더미 데이터 생성 ---
# 실제 이미지 입력 형태 (B, C, H, W)
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device) 
# 정규화된 나이 레이블 (0.0 ~ 1.0)
dummy_target = torch.rand(BATCH_SIZE).to(device)

# --- 더미 학습 스텝 실행 (통합 테스트) ---
print("\n--- Beta-NLL 모델 통합 테스트 시작 ---")
try:
    optimizer.zero_grad() 
    
    # 1. Forward Pass
    pred_alpha, pred_beta = model(dummy_input)
    
    # 2. Loss 계산 (손실 함수 입력 형태에 맞게 unsqueeze로 차원 확장)
    loss = criterion(pred_alpha.unsqueeze(1), pred_beta.unsqueeze(1), dummy_target.unsqueeze(1))
    
    # 3. Backward Pass & 가중치 업데이트
    loss.backward()
    optimizer.step()
    
    print("✅ 통합 테스트 성공! (Forward, Loss, Backward 모두 정상 작동)")
    print(f"최종 손실 값: {loss.item():.4f}")
    
except Exception as e:
    print(f"\n❌ 통합 테스트 중 오류 발생: {e}")
import torch
import torch.nn as nn
import torch.optim as optim
from models import BetaNLL_AgePredictor
from train import Beta_NLL_Loss


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon GPU (MPS)를 사용합니다.")
else:
    device = torch.device("cpu")
    print("⚠️ CPU를 사용합니다. GPU 가속을 확인하세요.")

# --- 모델 및 손실 함수 인스턴스화 ---
model = BetaNLL_AgePredictor(use_pretrained=True).to(device)
criterion = Beta_NLL_Loss(beta_reweight=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 더미 데이터 생성 ---
# B: 배치 크기, C: 채널(RGB), H, W: 이미지 크기 (ResNet18 기본 입력: 224x224)
BATCH_SIZE = 16
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device) 
# 정규화된 나이 레이블 (0.0 ~ 1.0)
dummy_target = torch.rand(BATCH_SIZE).to(device)

# --- 더미 학습 스텝 실행 (오류 확인) ---
try:
    optimizer.zero_grad() 
    
    # 1. Forward Pass
    pred_alpha, pred_beta = model(dummy_input)
    
    # 2. Loss 계산
    loss = criterion(pred_alpha, pred_beta, dummy_target)
    
    # 3. Backward Pass (역전파)
    loss.backward()
    
    # 4. Optimizer Step (가중치 업데이트)
    optimizer.step()
    
    print("\n✅ 모델 구조 및 손실 함수 통합 테스트 성공!")
    print(f"최종 손실 값: {loss.item():.4f}")
    print(f"예측 Alpha 평균: {pred_alpha.mean().item():.4f}")
    print(f"예측 Beta 평균: {pred_beta.mean().item():.4f}")

except Exception as e:
    print(f"\n❌ 통합 테스트 중 오류 발생: {e}")
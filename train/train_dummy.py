# train/train_dummy.py 수정 내용

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 1. 경로 추가: 최상위 폴더를 경로에 추가 (models, data 폴더 찾기 위함)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

# 2. 기존 import 대신 새로운 Dataset 클래스 import
from data.dataset import DeepfakeDataset  # <-- 이 줄을 추가합니다.
from models.model import BetaNLL_AgePredictor 
from models.losses import Beta_NLL_Loss

# --- 환경 설정 및 모델 인스턴스화 (유지) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = BetaNLL_AgePredictor(use_pretrained=True, freeze_backbone=True).to(device)
criterion = Beta_NLL_Loss(beta_reweight=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. 더미 데이터 생성 대신, 실제 데이터셋 로드 (수정)
# !!! 데이터셋의 최상위 폴더 경로를 정확히 지정해야 합니다 !!!
DATA_DIR = '../aidetector' # 예시 경로: train 폴더에서 두 단계 위 (깃 저장소 폴더)의 aidetector 폴더
BATCH_SIZE = 16

try:
    # DeepfakeDataset을 사용하여 train 분할의 데이터를 로드합니다.
    dataset = DeepfakeDataset(DATA_DIR, subset='train')
    
    # DataLoader 설정
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 첫 번째 배치 가져오기 (실제 이미지와 0/1 레이블)
    dummy_input, dummy_target = next(iter(dataloader))
    dummy_input = dummy_input.to(device)
    dummy_target = dummy_target.to(device).unsqueeze(1) # Loss 함수를 위해 차원 추가
    
except FileNotFoundError as e:
    print(f"❌ 데이터셋 로드 실패! 경로를 확인하세요: {e}")
    # 테스트 중단
    sys.exit(1)


# --- 학습 스텝 실행 (유지) ---
print("\n--- Deepfake Detector 통합 테스트 시작 ---")
try:
    optimizer.zero_grad() 
    
    # 1. Forward Pass
    pred_alpha, pred_beta = model(dummy_input)
    
    # 2. Loss 계산
    loss = criterion(pred_alpha.unsqueeze(1), pred_beta.unsqueeze(1), dummy_target)
    
    # 3. Backward Pass & 가중치 업데이트
    loss.backward()
    optimizer.step()
    
    print("✅ 통합 테스트 성공! (Forward, Loss, Backward 모두 정상 작동)")
    print(f"최종 손실 값: {loss.item():.4f}")
    
except Exception as e:
    print(f"\n❌ 통합 테스트 중 오류 발생: {e}")
# train/train_dummy.py

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# --- 1. 파일 경로 문제 해결 (모든 import 전에 실행되어야 함) ---
try:
    # 현재 파일의 상위 폴더(최상위 저장소 폴더)를 모듈 검색 경로에 추가합니다.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 
    
    # 이제 data/dataset.py와 models/를 정상적으로 가져옵니다.
    from data.dataset import DeepfakeDataset 
    from models.model import BetaNLL_AgePredictor 
    from models.losses import Beta_NLL_Loss

except ModuleNotFoundError as e:
    print("❌ 프로젝트 모듈 로드 실패! 'models' 또는 'data' 폴더를 찾을 수 없습니다.")
    print("오류 상세: 'data/dataset.py'와 'models/model.py' 파일이 올바른 위치에 있는지 확인하세요.")
    print(f"시스템 경로: {sys.path}")
    sys.exit(1)


# --- 2. 환경 및 모델 인스턴스화 ---

# Mac M-Chip (Apple Silicon) GPU 가속 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon GPU (MPS)를 사용합니다.")
else:
    device = torch.device("cpu")
    print("⚠️ CPU를 사용합니다. (성능 저하 가능)")

# 하이퍼파라미터
DATA_DIR = '../aidetector'  # train 폴더에서 두 단계 위로 이동하여 'aidetector' 폴더를 가리킵니다.
BATCH_SIZE = 16
BETA_REWEIGHT = 0.5 

# 모델 및 손실 함수 인스턴스화
model = BetaNLL_AgePredictor(use_pretrained=True, freeze_backbone=True).to(device)
criterion = Beta_NLL_Loss(beta_reweight=BETA_REWEIGHT).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# --- 3. 데이터셋 로드 및 학습 스텝 실행 ---
try:
    # DeepfakeDataset을 사용하여 'train' 분할의 데이터를 로드합니다.
    dataset = DeepfakeDataset(DATA_DIR, subset='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 첫 번째 배치 가져오기 (실제 이미지와 0/1 레이블)
    dummy_input, dummy_target = next(iter(dataloader))
    
    # 디바이스로 이동 및 차원 추가 (Loss 함수를 위해)
    dummy_input = dummy_input.to(device)
    dummy_target = dummy_target.to(device).unsqueeze(1) 
    
except FileNotFoundError as e:
    print("\n❌ 데이터셋 파일/폴더를 찾을 수 없습니다.")
    print(f"경로 오류: '{DATA_DIR}' 경로에 'real/train' 또는 'fake/train' 폴더가 있는지 확인하세요.")
    sys.exit(1)
except StopIteration:
    print("\n❌ 데이터로더에 데이터가 없습니다. 폴더에 jpg 파일이 비어있는지 확인하세요.")
    sys.exit(1)


# --- 4. 통합 테스트 실행 ---
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
    print(f"\n❌ 통합 테스트 중 알 수 없는 오류 발생: {e}")
    sys.exit(1)
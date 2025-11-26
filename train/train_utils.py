import torch
import os

# -------------------------------------------
# 1 epoch 동안 Deepfake 분류 모델 학습
# 입력  : 이미지 텐서
# 타깃  : label (real=0.0, fake=1.0)
# 출력  : 해당 epoch의 평균 loss
# -------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, ages in loader:
        # 데이터 GPU/MPS 이동
        imgs, ages = imgs.to(device), ages.to(device)

        optimizer.zero_grad()

        # 모델 forward
        preds = model(imgs)

        # Beta-NLL 모델은 (alpha, beta) 두 개 출력
        if isinstance(preds, tuple):
            alpha, beta = preds
            loss = criterion(alpha, beta, ages)
        else: # (혹시 나중에 MSE나 BCE 모델 추가할 때)
            loss = criterion(preds, ages)

        # backward + 가중치 업데이트
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# -------------------------------------------
# 검증 루프
# - gradient 계산 없음(@torch.no_grad)
# - Loss + MAE(예측 확률 vs 정답 레이블) 계산
# -------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0

    for imgs, ages in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)

        if isinstance(preds, tuple):
            alpha, beta = preds
            loss = criterion(alpha, beta, labels)
            # Beta 분포의 평균 = fake일 확률처럼 사용
            prob_fake = alpha / (alpha + beta)
        else:
            loss = criterion(preds, labels)
            prob_fake = preds  # (나중에 다른 모델 쓰면 확률 출력이라고 가정)

        # MAE: 예측 확률 vs 0/1 레이블 사이의 평균 절대 오차
        mae = torch.mean(torch.abs(prob_fake - labels))

        total_loss += loss.item()
        total_mae += mae.item()

    return total_loss / len(loader), total_mae / len(loader)
  
  
# -------------------------------------------
# 모델 체크포인트 저장 함수
# -------------------------------------------
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
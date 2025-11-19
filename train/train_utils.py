import torch
import os

# -------------------------------------------
# 1 epoch 동안의 학습을 수행하는 함수
# 모델 forward → loss 계산 → backward → optimizer.step
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
        else:
            loss = criterion(preds, ages)

        # backward + 가중치 업데이트
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# -------------------------------------------
# 모델 검증(validation)
# 학습과 다르게 gradient 계산 X
# Loss + MAE 평가
# -------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0

    for imgs, ages in loader:
        imgs, ages = imgs.to(device), ages.to(device)

        preds = model(imgs)

        if isinstance(preds, tuple):
            alpha, beta = preds
            loss = criterion(alpha, beta, ages)
            # Beta 분포의 평균 = alpha / (alpha + beta)
            pred_mean = alpha / (alpha + beta)
        else:
            loss = criterion(preds, ages)
            pred_mean = preds
            
        # MAE 계산
        mae = torch.mean(torch.abs(pred_mean - ages))

        total_loss += loss.item()
        total_mae += mae.item()

    return total_loss / len(loader), total_mae / len(loader)
  
  
# -------------------------------------------
# 모델 체크포인트 저장 함수
# -------------------------------------------
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
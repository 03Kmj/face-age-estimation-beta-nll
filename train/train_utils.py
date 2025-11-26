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

    for imgs, labels in loader:   # <-- labels 사용
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 모델은 alpha, beta 출력
        preds = model(imgs)

        # Beta-NLL 경우 (alpha, beta)
        if isinstance(preds, tuple):
            alpha, beta = preds
            loss = criterion(alpha, beta, labels)
        else:
            loss = criterion(preds, labels)

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

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)

        if isinstance(preds, tuple):
            alpha, beta = preds
            loss = criterion(alpha, beta, labels)

            # Beta분포 mean = alpha / (alpha + beta)
            pred_mean = alpha / (alpha + beta)
        else:
            loss = criterion(preds, labels)
            pred_mean = preds

        # MAE 계산 (0~1 사이)
        mae = torch.mean(torch.abs(pred_mean - labels))

        total_loss += loss.item()
        total_mae += mae.item()

    return total_loss / len(loader), total_mae / len(loader)
  
  
# -------------------------------------------
# 모델 체크포인트 저장 함수
# -------------------------------------------
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
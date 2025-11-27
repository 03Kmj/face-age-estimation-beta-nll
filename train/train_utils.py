import torch
import os

# -------------------------------------------
# 1 epoch 동안 Deepfake 분류 + 불확실성 학습
# -------------------------------------------
def train_one_epoch(model, loader, bce_criterion, beta_criterion, optimizer, device, beta_weight=0.1):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 1) Forward: 분류 logit + Beta 파라미터
        logits, alpha, beta = model(imgs)   # (B,), (B,), (B,)

        # 2) Binary classification loss (real/fake)
        bce_loss = bce_criterion(logits, labels)

        # 3) Beta-NLL loss용 target (0,1 극단 피하려고 약간 smooth)
        targets_beta = labels * 0.9 + 0.05    # 0 → 0.05, 1 → 0.95
        
        beta_loss = beta_criterion(alpha, beta, targets_beta)

        # 4) 총 loss = 분류 + 불확실성 가중합
        loss = bce_loss + beta_weight * beta_loss

        # 5) Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# -------------------------------------------
# 검증 루프
# - Loss + MAE(예측 확률 vs 정답 레이블) 계산
# -------------------------------------------
@torch.no_grad()
def validate(model, loader, bce_criterion, beta_criterion, device, beta_weight=0.1):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # 1) Forward
        logits, alpha, beta = model(imgs)

        # 2) BCE loss
        bce_loss = bce_criterion(logits, labels)

        targets_beta = labels * 0.9 + 0.05
        beta_loss = beta_criterion(alpha, beta, targets_beta)

        loss = bce_loss + beta_weight * beta_loss

        # 3) MAE는 "Fake일 확률 vs 정답" 기준
        prob_fake = torch.sigmoid(logits)     # (B,)
        mae = torch.mean(torch.abs(prob_fake - labels))

        total_loss += loss.item()
        total_mae  += mae.item()

    return total_loss / len(loader), total_mae / len(loader)



# -------------------------------------------
# 모델 체크포인트 저장 함수
# -------------------------------------------
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

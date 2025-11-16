import torch
import os
from torch.utils.tensorboard import SummaryWriter


# 1 epoch 학습
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0


    for imgs, ages in loader:
        imgs, ages = imgs.to(device), ages.to(device)


    optimizer.zero_grad()
    preds = model(imgs)


    if isinstance(preds, tuple): # BetaNLL
        alpha, beta = preds
        loss = criterion(alpha, beta, ages)
    else: # MSE
        loss = criterion(preds, ages)


    loss.backward()
    optimizer.step()


    total_loss += loss.item()


    return total_loss / len(loader)




# 검증
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
        pred_mean = alpha / (alpha + beta)
    else:
        loss = criterion(preds, ages)
        pred_mean = preds


    mae = torch.mean(torch.abs(pred_mean - ages))


    total_loss += loss.item()
    total_mae += mae.item()


    return total_loss / len(loader), total_mae / len(loader)




# 체크포인트
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
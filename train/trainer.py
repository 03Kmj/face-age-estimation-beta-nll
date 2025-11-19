import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 프로젝트 최상위 경로 import 위해 추가

from config import Config
from data.dataset import AgeDataset
from models.model import BetaNLL_AgePredictor
from models.losses import Beta_NLL_Loss
from train.train_utils import train_one_epoch, validate, save_checkpoint


# -------------------------------------------
# 커맨드라인 인자 설정
# python train/trainer.py --model beta_nll --epochs 30 --batch_size 32
# -------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["beta_nll", "mse"], default="beta_nll")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("2번째 버전 학습 시작12345")
    

    # --------------------------
    # Device 설정
    # --------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS)를 사용합니다.")
        print("🔥 Running trainer.py at:", __file__)

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 NVIDIA GPU(CUDA)를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU를 사용합니다. GPU 가속 없음.")

    # -------------------------------------------
    # Dataset 및 DataLoader 구성
    # 이미지 파일명에서 age를 읽는 AgeDataset 사용
    # -------------------------------------------
    train_dataset = AgeDataset(data_dir=Config.DATA_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # --------------------------
    # 모델 선택
    # --------------------------
    if args.model == "beta_nll":
        model = BetaNLL_AgePredictor().to(device)
        criterion = Beta_NLL_Loss()
    else:
        raise NotImplementedError("MSE 모델은 아직 생성하지 않았습니다.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard 설정
    writer = SummaryWriter(f"{Config.LOG_DIR}/{args.model}")

    best_loss = float("inf")

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, train_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val MAE:    {val_mae:.4f}")
        
        # TensorBoard 기록
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)

        # --------------------------
        # Best Model 저장
        # --------------------------
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = f"models/best_{args.model}_model.pth"
            save_checkpoint(model, save_path)

    writer.close()
    print("\n[Training Completed]")


if __name__ == "__main__":
    main()

# import argparse
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import sys, os, time
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 프로젝트 최상위 경로 import 위해 추가

# from config import Config
# from data.dataset import DeepfakeDataset
# from models.model import BetaNLL_AgePredictor
# from models.losses import Beta_NLL_Loss
# from train.train_utils import train_one_epoch, validate, save_checkpoint


# # -------------------------------------------
# # 커맨드라인 인자 설정
# # python train/trainer.py --model beta_nll --epochs 30 --batch_size 32
# # -------------------------------------------

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, choices=["beta_nll", "mse"], default="beta_nll")
#     parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
#     parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
#     parser.add_argument("--lr", type=float, default=Config.LR)
#     parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
#     return parser.parse_args()


# # ---------------------------
# # 시간 포맷 함수
# # ---------------------------
# def format_time(seconds):
#     m, s = divmod(int(seconds), 60)
#     h, m = divmod(m, 60)
#     if h > 0:
#         return f"{h}시간 {m}분 {s}초"
#     elif m > 0:
#         return f"{m}분 {s}초"
#     else:
#         return f"{s}초"


# def main():
#     args = parse_args()
    
#     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!딥페이크 탐지 시작(small 버전)-> .env 파일에서 경로 수정!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!딥페이크 탐지 시작!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    

#     # --------------------------
#     # Device 설정
#     # --------------------------
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("✅ Apple Silicon GPU (MPS)를 사용합니다.")
#         print("🔥 Running trainer.py at:", __file__)

#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("🚀 NVIDIA GPU(CUDA)를 사용합니다.")
#     else:
#         device = torch.device("cpu")
#         print("⚠️ CPU를 사용합니다. GPU 가속 없음.")

#     # -------------------------------------------
#     # Dataset 및 DataLoader 구성
#     # DeepfakeDataset: real / fake / subset(train/val/test)
#     # -------------------------------------------
#     train_dataset = DeepfakeDataset(Config.DATA_DIR, subset='train')
#     val_dataset   = DeepfakeDataset(Config.DATA_DIR, subset='val')

#     train_loader = DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers
#     )

#     val_loader = DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.num_workers
#     ) 

#     # --------------------------
#     # 모델 선택
#     # --------------------------
#     if args.model == "beta_nll":
#         model = BetaNLL_AgePredictor().to(device)
#         criterion = Beta_NLL_Loss()
#     else:
#         raise NotImplementedError("MSE 모델은 아직 생성하지 않았습니다.")

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # TensorBoard 설정
#     writer = SummaryWriter(f"{Config.LOG_DIR}/{args.model}")

#     best_loss = float("inf")

#     # --------------------------
#     # Training Loop
#     # --------------------------
    
#     total_epochs = args.epochs
#     epoch_times = []
#     for epoch in range(args.epochs):
#         print(f"\n[Epoch {epoch+1}/{args.epochs}]")

#         epoch_start = time.time()   # <-- 시작 시간 측정

#         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        
        
#         # --------------------------
#         # Epoch 종료 → 시간 계산
#         # --------------------------
#         epoch_time = time.time() - epoch_start
#         epoch_times.append(epoch_time)

#         print(f"⏱️ 이번 Epoch 소요 시간: {format_time(epoch_time)}")

#         # --------------------------
#         # ETA 계산 (평균 시간 기반)
#         # --------------------------
#         avg_time = sum(epoch_times) / len(epoch_times)
#         remaining_epochs = total_epochs - (epoch + 1)
#         eta = remaining_epochs * avg_time

#         print(f"🔮 예상 남은 시간(ETA): {format_time(eta)}")

#         # --------------------------
#         # 로그 출력
#         # --------------------------
#         print(f"Train Loss: {train_loss:.4f}")
#         print(f"Val Loss:   {val_loss:.4f}")
#         print(f"Val MAE:    {val_mae:.4f}")
        
#         # TensorBoard 기록
#         writer.add_scalar("Loss/train", train_loss, epoch)
#         writer.add_scalar("Loss/val", val_loss, epoch)
#         writer.add_scalar("MAE/val", val_mae, epoch)

#         # --------------------------
#         # Best Model 저장
#         # --------------------------
#         if val_loss < best_loss:
#             best_loss = val_loss
#             save_path = f"models/best_{args.model}_model.pth"
#             save_checkpoint(model, save_path)

#     writer.close()
#     print("\n[Training Completed]")


# if __name__ == "__main__":
#     main()


import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 프로젝트 최상위 경로 import 위해 추가

from config import Config
from data.dataset import DeepfakeDataset
from models.model import DeepfakeUncertaintyModel
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


# ---------------------------
# 시간 포맷 함수
# ---------------------------
def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}시간 {m}분 {s}초"
    elif m > 0:
        return f"{m}분 {s}초"
    else:
        return f"{s}초"


def main():
    args = parse_args()
    
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!딥페이크 탐지 시작(small 버전)-> .env 파일에서 경로 수정!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!딥페이크 탐지 시작!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    

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
    # DeepfakeDataset: real / fake / subset(train/val/test)
    # -------------------------------------------
    train_dataset = DeepfakeDataset(Config.DATA_DIR, subset='train')
    val_dataset   = DeepfakeDataset(Config.DATA_DIR, subset='val')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    ) 

    # --------------------------
    # 모델 선택
    # --------------------------
    if args.model == "beta_nll":
        model = DeepfakeUncertaintyModel().to(device)
        
        # 분류용 Loss (real/fake)
        bce_criterion  = nn.BCEWithLogitsLoss()
        # 불확실성(Beta)용 Loss
        beta_criterion = Beta_NLL_Loss()
        
        beta_weight = 0.01   # Beta-NLL을 얼마나 섞을지 가중치 (필요하면 튜닝)
    else:
        raise NotImplementedError("MSE 모델은 아직 생성하지 않았습니다.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard 설정
    writer = SummaryWriter(f"{Config.LOG_DIR}/{args.model}")

    best_loss = float("inf")

    # --------------------------
    # Training Loop
    # --------------------------
    
    total_epochs = args.epochs
    epoch_times = []
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        epoch_start = time.time()   # <-- 시작 시간 측정

        train_loss = train_one_epoch(model, train_loader, bce_criterion, beta_criterion, optimizer, device, beta_weight=beta_weight,)
        val_loss, val_mae = validate(model, val_loader, bce_criterion, beta_criterion, device, beta_weight=beta_weight,)
        
        
        
        # --------------------------
        # Epoch 종료 → 시간 계산
        # --------------------------
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"⏱️ 이번 Epoch 소요 시간: {format_time(epoch_time)}")

        # --------------------------
        # ETA 계산 (평균 시간 기반)
        # --------------------------
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - (epoch + 1)
        eta = remaining_epochs * avg_time

        print(f"🔮 예상 남은 시간(ETA): {format_time(eta)}")

        # --------------------------
        # 로그 출력
        # --------------------------
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

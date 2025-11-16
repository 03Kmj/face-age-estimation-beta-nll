import os
from dotenv import load_dotenv

# .env 파일 읽기
load_dotenv() 

class Config:
    # 데이터 경로
    CSV_PATH = os.getenv("CSV_PATH", "./data/age_labels.csv")
    IMG_DIR  = os.getenv("IMG_DIR", "./data/images/")

    # 데이터 설정
    IMG_SIZE = 224
    MAX_AGE  = 116.0

    # 학습 설정
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LR = 1e-4
    EPOCHS = 30
    CHECKPOINT_DIR = "./train/checkpoints"
    LOG_DIR = "./train/logs"
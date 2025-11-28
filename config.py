import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드 (.env파일에 다음과 같이 사진 폴더 경로 넣기 DATA_DIR="/Users/kang/Library/...")
load_dotenv() 

class Config:
    # -------------------------------------------
    # 데이터 설정
    # 이미지가 저장된 폴더 경로
    # -------------------------------------------
    DATA_DIR = os.getenv("DATA_DIR", "./aidetector")

    # -------------------------------------------
    # 학습 관련 하이퍼파라미터
    # -------------------------------------------
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LR = 1e-4
    EPOCHS = 30

    # -------------------------------------------
    # 로그 및 모델 저장 경로
    # -------------------------------------------
    LOG_DIR = "./train/logs"
    CHECKPOINT_DIR = "./train/checkpoints"
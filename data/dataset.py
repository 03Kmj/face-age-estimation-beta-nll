import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import Config


class UTKFaceDataset(Dataset):

    
    def __init__(self, csv_path=None, img_dir=None, img_size=None, transform=None):
        # # 디버깅용
        # self.cnt = 1

        self.csv_path = csv_path or Config.CSV_PATH
        self.img_dir = img_dir or Config.IMG_DIR
        self.img_size = img_size or Config.IMG_SIZE

        self.df = pd.read_csv(self.csv_path, encoding="cp949")


        # Transform
        self.transform = transform or T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[Dataset] Loaded CSV: {self.csv_path}")
        print(f"[Dataset] IMG_DIR: {self.img_dir}")
        print(f"[Dataset] Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)


    def find_augmented_image(self, age, gender, race, original_id):
        """
        증강된 파일 이름 찾기:
        aug_*_age_gender_race_originalID.jpg.chip.jpg
        """

        original_id = original_id.replace("\\", "/").split("/")[-1]  # 경로에서 파일명만 추출
        original_id = original_id.split("_")[-1]  # "11_0_0_20170110220453002.jpg.chip.jpg" → "20170110220453002.jpg.chip.jpg"

        for fname in os.listdir(self.img_dir):

            if not fname.startswith("aug_"):
                continue

            parts = fname.split("_")
            # parts = ['aug', A, AGE, GENDER, RACE, ORIGINALID... ]

            if len(parts) < 6:
                continue

            # 매칭 조건
            if (
                parts[2] == str(age) and
                parts[3] == str(gender) and
                parts[4] == str(race) and
                parts[5].startswith(original_id.split(".")[0])  # ID 비교 (확장자 제외)
            ):
                return os.path.join(self.img_dir, fname)

        return None


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        age = int(row["age"])
        gender = int(row["gender"])
        race = int(row["race"])
        path_raw = row["path"]

        # 원본 ID 추출
        original_id = os.path.basename(path_raw)

        # 증강 이미지 파일 찾기
        img_path = self.find_augmented_image(age, gender, race, original_id)
        
        # # 디버깅용 출력
        # print(img_path)
        # print(self.cnt)
        # self.cnt += 1

        if img_path is None:
            raise FileNotFoundError(f"증강 이미지 찾을 수 없음: age={age}, g={gender}, r={race}, id={original_id}")

        # 이미지 로드
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 나이 정규화
        age_norm = age / Config.MAX_AGE

        return img, torch.tensor(age_norm, dtype=torch.float32)

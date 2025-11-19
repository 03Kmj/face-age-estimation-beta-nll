import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple

# --- 전역 상수 정의 ---
# 나이 범위 (대부분의 공개 데이터셋은 0세부터 100세까지를 커버합니다)
MIN_AGE = 0
MAX_AGE = 100
# ResNet18 백본에 맞는 표준 입력 크기
IMG_SIZE = 224

def normalize_age(actual_age: int) -> float:
    """실제 나이(정수)를 [0, 1] 범위로 정규화합니다."""
    # 나이가 정의된 범위를 벗어나지 않도록 클리핑합니다.
    age_clipped = max(MIN_AGE, min(MAX_AGE, actual_age))
    
    # 정규화 공식 적용: (나이 - 최소 나이) / (최대 나이 - 최소 나이)
    normalized = (age_clipped - MIN_AGE) / (MAX_AGE - MIN_AGE)
    
    return normalized

def extract_and_normalize_age(filename: str) -> float:
    """
    파일 이름에서 나이를 추출하고 [0, 1]로 정규화합니다.
    (예시 파일명: aug_0_1_0_0_...chip.jpg 에서 두 번째 '_' 뒤의 '1'을 나이로 가정)
    """
    try:
        # 파일명을 '_' 기준으로 분리합니다.
        parts = filename.split('_')
        
        # 나이 레이블이 두 번째 인덱스(세 번째 요소)에 있다고 가정합니다.
        # 예: parts[2] = '1', '60' 등
        if len(parts) > 2:
            actual_age = int(parts[2])
            return normalize_age(actual_age)
        
    except ValueError:
        # 파일 이름에서 숫자를 추출하지 못했을 경우 (오류 방지)
        print(f"경고: 파일 {filename}에서 나이를 추출할 수 없습니다. 0.5(50세)로 가정합니다.")
        return 0.5
    except IndexError:
        # 파일 이름 구조가 예상과 다를 경우 (오류 방지)
        print(f"경고: 파일 {filename}의 구조가 잘못되었습니다. 0.5(50세)로 가정합니다.")
        return 0.5
    
    return 0.5 # 오류 발생 시 기본값 반환

class AgeDataset(Dataset):
    """얼굴 이미지 나이 예측을 위한 커스텀 PyTorch Dataset 클래스"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        
        # 이미지 파일 목록 로드 및 레이블 추출
        for filename in os.listdir(data_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.chip.jpg')):
                # 1. 파일 경로 저장
                img_path = os.path.join(data_dir, filename)
                self.image_paths.append(img_path)
                
                # 2. 나이 추출 및 정규화
                normalized_age = extract_and_normalize_age(filename)
                self.labels.append(normalized_age)

        # ResNet에 맞는 표준 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)), # 크기 조정
            transforms.ToTensor(), # HWC -> CHW, [0, 255] -> [0, 1] 정규화
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 평균 및 표준편차
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """데이터셋의 전체 샘플 개수를 반환합니다."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 이미지 텐서와 정규화된 나이 텐서를 반환합니다.
        """
        # 1. 이미지 로드 및 변환
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # 2. 레이블 텐서 생성
        normalized_age = self.labels[idx]
        # 모델이 요구하는 텐서 형태로 변환 (1.0은 float 타입)
        label_tensor = torch.tensor(normalized_age, dtype=torch.float32)
        
        return image_tensor, label_tensor

if __name__ == '__main__':
    # --- 테스트 코드 (실제 실행 시 data_dir을 수정해야 합니다) ---
    print("--- AgeDataset 테스트 시작 ---")
    
    # !!! 중요: 'data' 폴더의 실제 경로로 수정하세요 !!!
    test_data_dir = './' 
    
    try:
        # 1. Dataset 인스턴스화
        dataset = AgeDataset(test_data_dir)
        print(f"데이터셋에 총 {len(dataset)}개의 파일이 로드되었습니다.")
        
        # 2. 첫 번째 샘플 가져오기
        if len(dataset) > 0:
            first_image, first_label = dataset[0]
            print(f"첫 번째 이미지 텐서 크기: {first_image.shape}") # 예상: torch.Size([3, 224, 224])
            print(f"첫 번째 레이블 (정규화된 나이): {first_label.item():.4f}") # 예상: 0.xx (0~1 사이 값)
            print(f"첫 번째 레이블 (실제 나이 가정): {first_label.item() * (MAX_AGE - MIN_AGE) + MIN_AGE:.2f}세")
        
    except FileNotFoundError:
        print(f"오류: 폴더 {test_data_dir}를 찾을 수 없습니다. 경로를 확인하세요.")
    except Exception as e:
        print(f"테스트 중 알 수 없는 오류 발생: {e}")
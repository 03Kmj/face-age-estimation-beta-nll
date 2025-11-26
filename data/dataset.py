import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple

# --- 전역 상수 정의 ---
IMG_SIZE = 224 # ResNet18 백본에 맞는 표준 입력 크기

class DeepfakeDataset(Dataset):
    """
    AI 생성 이미지 탐지를 위한 커스텀 PyTorch Dataset 클래스.
    데이터는 {data_dir}/real/{subset}/ 및 {data_dir}/fake/{subset}/ 구조를 따릅니다.
    """

    def __init__(self, data_dir: str, subset: str = 'train'):
        """
        :param data_dir: 데이터셋의 최상위 폴더 경로 (예: 'aidetector')
        :param subset: 로드할 데이터셋 분할 (필수: 'train', 'val', 'test' 중 하나)
        """
        if subset not in ['train', 'val', 'test']:
            raise ValueError("subset은 'train', 'val', 'test' 중 하나여야 합니다.")
            
        self.data_dir = data_dir
        self.subset = subset
        self.image_paths = []
        self.labels = []
        
        # --- 레이블링 및 파일 탐색 ---
        # 0.0: Real (진짜), 1.0: Fake (가짜)
        
        categories = {'real': 0.0, 'fake': 1.0}
        
        for category, label in categories.items():
            # 최종 이미지 폴더 경로: data_dir / category (e.g., real) / subset (e.g., train)
            final_img_dir = os.path.join(data_dir, category, subset)
            
            if os.path.exists(final_img_dir):
                for filename in os.listdir(final_img_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.chip.jpg')):
                        self.image_paths.append(os.path.join(final_img_dir, filename))
                        self.labels.append(label)
            
        if not self.image_paths:
             raise FileNotFoundError(f"데이터셋 폴더({data_dir})의 {self.subset} 분할에 이미지가 없습니다.")
           
           
        print(f"[{self.subset}] 총 이미지 개수:", len(self.image_paths))
        print(f"[{self.subset}] real 개수:", self.labels.count(0.0))
        print(f"[{self.subset}] fake 개수:", self.labels.count(1.0))
        print("-" * 40)


        # ResNet에 맞는 표준 이미지 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """데이터셋의 전체 샘플 개수를 반환합니다."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 이미지 텐서와 레이블 텐서(0.0 또는 1.0)를 반환합니다.
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            # 파일이 손상되었거나 접근할 수 없을 경우 다음 샘플 로드 (재귀)
            # print(f"오류 발생: 이미지 로드 실패 - {img_path}. 다음 샘플로 넘어갑니다.")
            return self.__getitem__((idx + 1) % len(self)) 
        
        label_value = self.labels[idx]
        label_tensor = torch.tensor(label_value, dtype=torch.float32)
        
        return image_tensor, label_tensor

if __name__ == '__main__':
    # --- 테스트 코드 (실제 실행 시 경로를 수정해야 합니다) ---
    print("--- DeepfakeDataset 테스트 시작 (train 분할 로드) ---")
    
    # !!! 이 경로를 실제 'aidetector' 폴더 경로로 수정하세요 !!!
    test_data_dir = '../data_files/aidetector' 
    
    try:
        # Train 데이터셋 로드 테스트
        train_dataset = DeepfakeDataset(test_data_dir, subset='train')
        print(f"Train 데이터셋 총 {len(train_dataset)}개 파일 로드.")
        
        # Test 데이터셋 로드 테스트
        test_dataset = DeepfakeDataset(test_data_dir, subset='test')
        print(f"Test 데이터셋 총 {len(test_dataset)}개 파일 로드.")

    except FileNotFoundError as e:
        print(f"❌ 오류: 경로를 확인하세요: {e}")
    except Exception as e:
        print(f"❌ 테스트 중 알 수 없는 오류 발생: {e}")

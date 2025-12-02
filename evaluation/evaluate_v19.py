import torch
import pandas as pd
import numpy as np
import os
import sys
import shutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from models.model import DeepfakeUncertaintyModel

class StrictCSVDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        print(f"📄 [핵심] 정답지 로딩 중: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
            if 'split' in df.columns:
                self.data = df[df['split'] == 'test'].reset_index(drop=True)
            else:
                self.data = df
            print(f"👉 분석 대상: {len(self.data)}장")
        except:
            self.data = pd.DataFrame()

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # [수정 완료] 라벨이 'real' 글자로 되어 있어도 숫자로 변환!
        raw_label = str(row.get('label', '0')).lower().strip()
        if raw_label == 'real' or raw_label == '0': label = 0
        elif raw_label == 'fake' or raw_label == '1': label = 1
        else: label = 0 # 모르면 Real
        
        raw_path = str(row.get('filepath', row.get('path', '')))
        filename = os.path.basename(raw_path)
        folder_name = 'fake' if label == 1 else 'real'
        
        possible_paths = [
            os.path.join(self.data_root, folder_name, 'test', filename),
            os.path.join(self.data_root, folder_name, 'train', filename),
            os.path.join(self.data_root, folder_name, 'val', filename),
            os.path.join(self.data_root, folder_name, filename)
        ]
        img_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if img_path is None: return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32), "None"

        try: image = Image.open(img_path).convert('RGB')
        except: return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32), "None"

        if self.transform: image = self.transform(image)
        else: image = transforms.ToTensor()(image)
        
        return image, torch.tensor(label, dtype=torch.float32), img_path

def run():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 [평가 시스템] 가동")

    data_root = './data' 
    csv_path = './data/metadata.csv' 
    model_path = './models/best_beta_nll_model.pth'

    model = DeepfakeUncertaintyModel(use_pretrained=False).to(device)
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            if 'state_dict' in ckpt: model.load_state_dict(ckpt['state_dict'])
            else: model.load_state_dict(ckpt)
            print("✅ 학습된 모델 장착 완료!")
        except: print("⚠️ 모델 로딩 실패 -> 랜덤 모드")
    else: print("⚠️ 모델 파일 없음 -> 랜덤 모드")
    model.eval()

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = StrictCSVDataset(csv_path, data_root, transform=tf)
    
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    results = []
    
    print("⚡ 심층 분석 중...")
    with torch.no_grad():
        for images, labels, paths in tqdm(loader):
            if images.sum() == 0: continue
            images = images.to(device)
            logit, alpha, beta = model(images)
            pred_probs = torch.sigmoid(logit).cpu().numpy()
            uncertainties = ((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))).cpu().numpy()
            labels = labels.cpu().numpy()
            
            for i in range(len(images)):
                pred_label = 1 if pred_probs[i] > 0.5 else 0
                is_correct = (pred_label == labels[i])
                
                results.append({
                    'path': paths[i],
                    'true_label': labels[i],
                    'pred_label': pred_label,
                    'is_correct': 'Correct (정답)' if is_correct else 'Wrong (오답)',
                    'uncertainty': uncertainties[i],
                    'prob': pred_probs[i]
                })

    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("📊 [최종 심화 성적표]")
        print("="*60)
        
        acc = accuracy_score(df['true_label'], df['pred_label'])
        print(f"\n🏆 전체 정확도: {acc:.4f}")
        print(f"\n⚖️  Real/Fake 불확실성 차이:")
        print(df.groupby('true_label')['uncertainty'].mean().rename({0.0:'Real', 1.0:'Fake'}))

        print("\n🧠 [정답 vs 오답 불확실성 비교]")
        print(df.groupby('is_correct')['uncertainty'].mean())

        print("\n📸 [AI가 가장 헷갈려한 사진 Top 10]")
        top_confusion = df.sort_values(by='uncertainty', ascending=False).head(10)
        for idx, row in top_confusion.iterrows():
            fname = os.path.basename(str(row['path']))
            ans = 'Fake' if row['true_label']==1 else 'Real'
            pred = 'Fake' if row['pred_label']==1 else 'Real'
            print(f"   [{idx+1}줄] {fname} | 정답: {ans} | 예측: {pred} | 불확실성: {row['uncertainty']:.6f}")
        
        print("\n✅ 분석 완료!")
    else:
        print("\n❌ 결과 없음.")

if __name__ == '__main__':
    run()
import torch
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from models.model import DeepfakeUncertaintyModel

def analyze_folder(model, device, folder_path, label_name):
    if not os.path.exists(folder_path):
        print(f"❌ 폴더 없음: {folder_path}")
        return []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📂 '{label_name}' 분석 중... ({len(files)}장)")
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    results = []

    with torch.no_grad():
        for f in files:
            img_path = os.path.join(folder_path, f)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = tf(img).unsqueeze(0).to(device)
                
                logit, alpha, beta = model(img_tensor)
                
                prob = torch.sigmoid(logit).item()
                unc = ((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))).item()
                
                # 예측 (0.5 초과면 Fake)
                pred_label = 'Fake' if prob > 0.5 else 'Real'
                
                # 채점 (이 폴더의 사진들은 무조건 Fake니까, Fake라고 해야 정답)
                is_correct = 'O' if pred_label == 'Fake' else 'X'

                results.append({
                    'Type': label_name,
                    'Filename': f,
                    'Prediction': pred_label,
                    'Result': is_correct, # 채점 결과 추가
                    'Prob': prob,
                    'Uncertainty': unc
                })
            except Exception as e:
                print(f"⚠️ 에러({f}): {e}")

    return results

def run():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 [딥페이크 유형별 상세 분석 V3] 가동 (Device: {device})")

    model_path = './models/best_beta_nll_model.pth'
    model = DeepfakeUncertaintyModel(use_pretrained=False).to(device)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if 'state_dict' in ckpt: model.load_state_dict(ckpt['state_dict'])
        else: model.load_state_dict(ckpt)
        print("✅ 학습된 모델 장착 완료!")
    else:
        print("⚠️ 모델 파일 없음! (랜덤 테스트)")
    model.eval()

    data_root = './data/special_test'
    
    res_fully = analyze_folder(model, device, os.path.join(data_root, 'fully'), 'Fully Generated')
    res_partial = analyze_folder(model, device, os.path.join(data_root, 'partial'), 'Partially Manipulated')

    all_results = res_fully + res_partial
    
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # 1. 요약표 출력
        print("\n" + "="*60)
        print("📊 [1. 유형별 불확실성 평균]")
        print("="*60)
        summary = df.groupby('Type')[['Uncertainty', 'Prob']].mean().sort_values(by='Uncertainty', ascending=False)
        print(summary)
        
        # 2. 개별 상세 리스트 출력 (기배님이 원하신 것!)
        print("\n" + "="*60)
        print("🔍 [2. 개별 사진 채점표]")
        print("="*60)
        # 보기 좋게 컬럼 순서 정리
        print(df[['Type', 'Filename', 'Prediction', 'Result', 'Uncertainty']].to_string(index=False))

        # 3. 최종 결론
        print("\n" + "="*60)
        print("🏆 [최종 결론]")
        print("="*60)
        try:
            unc_fully = df[df['Type'] == 'Fully Generated']['Uncertainty'].mean()
            unc_partial = df[df['Type'] == 'Partially Manipulated']['Uncertainty'].mean()
            
            if np.isnan(unc_fully): unc_fully = 0
            if np.isnan(unc_partial): unc_partial = 0

            if unc_partial > unc_fully:
                diff = unc_partial / (unc_fully + 1e-9)
                print(f"👉 결론: '부분 조작(Partially)'이 약 {diff:.1f}배 더 불확실합니다.")
            elif unc_fully > unc_partial:
                diff = unc_fully / (unc_partial + 1e-9)
                print(f"👉 결론: '완전 생성(Fully)'이 약 {diff:.1f}배 더 불확실합니다.")
            else:
                print("👉 결론: 비슷합니다.")
        except: pass

        print("\n✅ 분석 완료!")
    else:
        print("\n❌ 데이터가 없습니다.")

if __name__ == '__main__':
    run()
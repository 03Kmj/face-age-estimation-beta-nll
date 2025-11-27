import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys, os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import DeepfakeUncertaintyModel

# -------------------------
# 이미지 전처리 (dataset.py와 동일)
# -------------------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------
# Beta Variance 계산 함수
# -------------------------
def beta_variance(alpha, beta):
    return (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))


# -------------------------
# Inference 함수
# -------------------------
@torch.no_grad()
def run_inference(img_path, model_path="models/best_beta_nll_model.pth", device="cpu"):

    # 1) 모델 로드
    model = DeepfakeUncertaintyModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) 이미지 불러오기
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # (1,3,224,224)

    # 3) Forward
    logit, alpha, beta = model(img)

    # 4) 확률 계산
    fake_prob = torch.sigmoid(logit).item()
    real_prob = 1 - fake_prob

    alpha_val = alpha.item()
    beta_val = beta.item()

    # 5) 불확실성 계산 (Beta 분포 variance)
    uncert = beta_variance(alpha_val, beta_val)

    # 6) 예측 레이블
    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"

    # 7) 출력
    print("===== Deepfake Detection Result =====")
    print(f"fake_probability: {fake_prob:.4f}")
    print(f"real_probability: {real_prob:.4f}")
    print(f"alpha: {alpha_val:.2f}")
    print(f"beta: {beta_val:.2f}")
    print(f"uncertainty: {uncert:.6f}")
    print(f"prediction: {prediction}")

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "alpha": alpha_val,
        "beta": beta_val,
        "uncertainty": uncert,
        "prediction": prediction
    }


# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="이미지 파일 경로")
    args = parser.parse_args()

    run_inference(args.img)

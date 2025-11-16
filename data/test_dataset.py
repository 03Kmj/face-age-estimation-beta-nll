from data.dataset import UTKFaceDataset

csv_path = "./data/age_labels.csv"
img_dir = "./data/images"

dataset = UTKFaceDataset(csv_path, img_dir)

print("총 데이터 개수:", len(dataset))
img, age = dataset[0]

print("이미지 shape:", img.shape)
print("age:", age)
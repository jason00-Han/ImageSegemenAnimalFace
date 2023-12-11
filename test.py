import io
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from app import get_embedding, get_prediction, batch_predict  # app.py 파일에서 필요한 함수들을 임포트

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 이미지 변환 설정
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image_as_bytes(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return IMG_TRANSFORM(image).unsqueeze(0)

# 이미지 파일 경로 리스트
image_paths = ['test_image.jpg', 'test_image2.jpg', 'test_image3.jpg']  # 여기에 테스트하려는 이미지 파일 경로를 입력

# 이미지를 바이트로 변환하고 예측 수행
predictions = []
for image_path in image_paths:
    img_bytes = load_image_as_bytes(image_path)
    transformed_image = transform_image(img_bytes).to(device)
    embedding = get_embedding(transformed_image)  # 이미 변환된 이미지를 사용
    prediction = get_prediction(embedding)
    predictions.append(prediction)


# batch_predict 함수를 사용하여 예측
img_bytes_list = [load_image_as_bytes(img_path) for img_path in image_paths]
batch_preds = batch_predict(img_bytes_list)

# 결과 출력
print("Individual Predictions:")
for pred in predictions:
    print(pred)

print("\nBatch Predictions:")
print(batch_preds)
print("Shape of Batch Predictions:", batch_preds.shape)

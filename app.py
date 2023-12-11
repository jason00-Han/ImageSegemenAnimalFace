from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import torch.nn as nn 
import torchvision.transforms as transforms
from flask_cors import CORS
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import label2rgb
import base64
import numpy as np


app = Flask(__name__)
CORS(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 이미지 변환 설정
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화 추가
])


def batch_predict_lime(images):
    preds = []
    for img_array in images:
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        embedding = get_embedding(img_bytes.getvalue())
        pred = get_prediction_lime(embedding)
        preds.append(pred)
    return np.array(preds)




# LIME 설명 생성 함수
def explain_with_lime(image_bytes, num_classes):
    # 이미지 변환
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # 슈퍼픽셀 생성을 위한 세그멘테이션 알고리즘 설정
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

    # LIME 설명자 초기화
    explainer = lime_image.LimeImageExplainer()

    # LIME 설명 생성
    explanation = explainer.explain_instance(np.array(image), 
                                             batch_predict_lime, 
                                             top_labels=num_classes, 
                                             hide_color=0, 
                                             num_samples=1000, 
                                             segmentation_fn=segmenter)

    # 설명 시각화
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    img_boundry = label2rgb(mask, temp, bg_label=0)

    # 시각화된 설명을 이미지로 변환
    plt.imshow(img_boundry)
    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return encoded_img

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # 이미지가 RGBA 포맷인 경우 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    transformed_image = IMG_TRANSFORM(image)
    transformed_image = transformed_image.unsqueeze(0)
    # 변환된 이미지 확인을 위한 출력
    #transformed_image_unsq = transformed_image.unsqueeze(0)
    #print("Transformed Image Size:", transformed_image.size())
    return transformed_image


class ModifiedClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ModifiedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)  # 상태 사전에 맞추어 크기 조정
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)  # fc3 레이어 추가
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  # fc3 레이어를 포함한 순전파
        return x


# 분류 모델 인스턴스 생성 및 상태 사전 로드
# 임베딩 크기를 알아야 정확한 input_size를 설정할 수 있습니다.
# InceptionResnetV1의 경우 일반적으로 512 또는 2048입니다.
input_size = 512  # InceptionResnetV1 출력 크기에 따라 변경
hidden_size = 256
num_classes = 2

# 이미지 임베딩 모델 (InceptionResnetV1) 인스턴스 생성
e_model = InceptionResnetV1(pretrained=None)
# 이 경우, 로드하는 상태 사전의 logits.weight 크기는 [2, 1792]입니다.
e_model.logits = nn.Linear(1792, num_classes)  # num_classes는 출력 클래스 수입니다.
# 저장된 상태 사전 로드
e_model.load_state_dict(torch.load('animal_model_state_dict_re.pth', map_location=device))
# 모델을 평가 모드로 설정하고 디바이스로 이동
e_model = e_model.to(device).eval()


model = ModifiedClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model = model.to(device).eval()

#이미지에서 임베딩 벡터 추출
def get_embedding(image_bytes):
    tensor = transform_image(image_bytes).to(device)
    with torch.no_grad():
        embedding = e_model(tensor)
        embedding = embedding.view(embedding.size(0), -1)  # 평탄화
    return embedding
    
def get_prediction_lime(embedding):
    with torch.no_grad():
        outputs = model(embedding)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 이진 분류에서는 두 클래스에 대한 확률을 반환합니다.
        return probabilities.numpy()[0]  # 첫 번째 샘플의 확률 반환

# 임베딩을 사용하여 예측 수행
def get_prediction(embedding):
    with torch.no_grad():
        outputs = model(embedding)
        _, predicted = outputs.max(1)
    return predicted.item()

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')

@app.route('/lime', methods=['POST'])
def lime():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'})
    file = request.files['file']
    img_bytes = file.read()
    embedding = get_embedding(img_bytes)
    prediction = get_prediction_lime(embedding)
    # NumPy 배열을 리스트로 변환
    prediction_list = prediction.tolist()

    # LIME 설명 생성
    lime_explanation = explain_with_lime(img_bytes, num_classes)

    # 응답에 LIME 설명 포함
    response = {
        'class_id': prediction_list,
        'lime_explanation': lime_explanation
    }

    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'})
    file = request.files['file']
    img_bytes = file.read()
    embedding = get_embedding(img_bytes)
    prediction = get_prediction(embedding)
    return jsonify({'class_id': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
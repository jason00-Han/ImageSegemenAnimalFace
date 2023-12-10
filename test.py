import torch

# 모델을 CPU로 로드
state_dict = torch.load('animal_model_state_dict.pth', map_location=torch.device('cpu'))
print(state_dict.keys())

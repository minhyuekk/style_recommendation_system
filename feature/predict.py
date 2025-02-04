import torch
import json
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn


class ResNetAttributeModel(nn.Module):
    def __init__(self, num_attributes):
        super(ResNetAttributeModel, self).__init__()
        # ResNet50
        self.resnet = models.resnet50(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_attributes)

    def forward(self, x):
        return self.resnet(x)


resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet 정규화
])


def load_attribute_names(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    attribute_names = {item["id"]: item["name"] for item in data["attributes"]}
    return attribute_names


def predict_image(model, image_path, attribute_names, device):
    model.eval()  # 모델을 평가 모드로 설정
    image = Image.open(image_path).convert("RGB")

    input_tensor = resnet_preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.sigmoid(outputs).squeeze(0)  # sigmoid

    # 임계값
    threshold = 0.5
    predicted_indices = (preds >= threshold).nonzero(as_tuple=True)[0].tolist()

    print(f"Predicted attribute indices: {predicted_indices}")
    print("Predicted attributes:")
    for idx in predicted_indices:
        if idx in attribute_names:
            print(f"- {attribute_names[idx]}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../feature_extractor/weights/fashion_attribute_model.pth"
    test_image_path = "../park_bo_gum.jpg"
    label_file = "../feature_extractor/data/labels/attributes_train2020.json"

    # 속성 이름 로드
    attribute_names = load_attribute_names(label_file)

    num_attributes = len(attribute_names)  # 속성 수 계산
    model = ResNetAttributeModel(num_attributes=num_attributes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 이미지 예측
    predict_image(model, test_image_path, attribute_names, device)

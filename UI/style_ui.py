import streamlit as st
from PIL import Image
import torch
import clip
import json
import cv2
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import numpy as np


class ResNetAttributeModel(nn.Module):
    def __init__(self, num_attributes):
        super(ResNetAttributeModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_attributes)

    def forward(self, x):
        return self.resnet(x)


resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


# BMI
def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)


def body_type(bmi):
    if bmi < 18.5:
        return "Slim"
    elif 18.5 <= bmi < 24.9:
        return "Balanced"
    elif 25 <= bmi < 29.9:
        return "Chubby"
    else:
        return "Fuller"


def check_style_match(body, attributes):
    style_rules = {
        "Slim": ["Patterned clothes", "Layered styles", "Skinny jeans"],
        "Balanced": ["Fitted clothes", "High-waist pants", "Bodycon dress"],
        "Chubby": ["Dark colors", "A-line dresses", "Vertical stripes"],
        "Fuller": ["Loose tops", "Draped styles", "High-waist pants"]
    }

    matched_styles = []
    for attribute in attributes:
        if attribute in style_rules[body]:
            matched_styles.append(attribute)

    if matched_styles:
        return True, matched_styles
    else:
        return False, style_rules[body]


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLO
    yolo_model = YOLO("../YOLO/weights/deepfashion2_yolov8s-seg.pt")

    # ResNet
    resnet_model_path = "../feature_extractor/weights/fashion_attribute_model.pth"
    label_path = "../feature_extractor/data/labels/attributes_train2020.json"
    with open(label_path, "r", encoding="utf-8") as f:
        attribute_names = {item["id"]: item["name"] for item in json.load(f)["attributes"]}

    resnet_model = ResNetAttributeModel(num_attributes=len(attribute_names))
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
    resnet_model.to(device)
    resnet_model.eval()

    # CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device)

    return yolo_model, resnet_model, clip_model, attribute_names, device


def classify_attributes(model, image, bbox, attribute_names, device):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = image.crop((x1, y1, x2, y2))
    input_tensor = resnet_preprocess(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.sigmoid(outputs).squeeze(0)

    threshold = 0.5
    detected_attributes = [
        attribute_names[idx] for idx, score in enumerate(preds) if score >= threshold and idx in attribute_names
    ]
    return detected_attributes


def match_style(clip_model, image, detected_attributes, style_templates, device):
    description = ", ".join(detected_attributes)
    text_descriptions = [f"This outfit has {description}."] + style_templates
    text_inputs = clip.tokenize(text_descriptions).to(device)

    input_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(input_tensor)
        text_features = clip_model.encode_text(text_inputs)
        similarities = torch.cosine_similarity(image_features, text_features)
        predicted_style_idx = similarities[1:].argmax()

    return style_templates[predicted_style_idx]


# --- Streamlit UI ---
def main():
    st.title("Fashion Style Matcher 👗🔶️")
    st.write("Upload an image to analyze and match the style of your outfit!")

    height = st.number_input("Height (cm):", min_value=100, max_value=250, step=10)
    weight = st.number_input("Weight (kg):", min_value=30, max_value=200, step=5)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image and height > 0 and weight > 0 is not None:
        bmi = calculate_bmi(height, weight)
        body = body_type(bmi)
        st.write(f"**Your estimated body type based on BMI is:** {body}")

        image = Image.open(uploaded_image).convert("RGB")

        # Model
        yolo_model, resnet_model, clip_model, attribute_names, device = load_models()

        style_templates = [
            # 캐주얼 스타일
            "This outfit is casual and comfortable.",
            "This outfit is perfect for daily casual wear.",
            "This outfit is relaxed and ideal for leisure time.",
            "This outfit combines comfort with a touch of style.",

            # 포멀 스타일
            "This outfit is formal and professional.",
            "This outfit is tailored for office wear or business meetings.",
            "This outfit represents an elegant business style.",
            "This outfit is perfect for a formal dinner or corporate event.",

            # 스트릿웨어 스타일
            "This outfit represents streetwear fashion.",
            "This outfit showcases urban street style.",
            "This outfit blends contemporary and edgy street fashion.",
            "This outfit is bold and inspired by modern street culture.",

            # 스포츠 스타일
            "This outfit is sporty and athletic.",
            "This outfit is designed for workouts and sports activities.",
            "This outfit combines functionality and style for activewear.",
            "This outfit is comfortable and suitable for outdoor activities.",

            # 우아한 스타일
            "This outfit is elegant and sophisticated.",
            "This outfit reflects a timeless and luxurious style.",
            "This outfit is perfect for formal occasions and upscale events.",
            "This outfit exudes class and refined taste.",

            # 빈티지 스타일
            "This outfit features a vintage and retro aesthetic.",
            "This outfit draws inspiration from classic fashion trends.",
            "This outfit showcases old-school charm and timeless appeal.",
            "This outfit reflects a love for retro patterns and colors.",

            # 보헤미안 스타일
            "This outfit represents a bohemian and free-spirited look.",
            "This outfit features flowing fabrics and natural patterns.",
            "This outfit reflects a relaxed and artistic boho style.",
            "This outfit embraces earthy tones and eclectic designs.",

            # 미니멀리즘 스타일
            "This outfit reflects a minimal and clean aesthetic.",
            "This outfit features simple lines and a neutral palette.",
            "This outfit embraces a less-is-more fashion philosophy.",
            "This outfit highlights elegance through simplicity.",

            # 럭셔리 스타일
            "This outfit represents high-end luxury fashion.",
            "This outfit showcases designer pieces and premium fabrics.",
            "This outfit reflects opulence and exclusivity.",
            "This outfit is perfect for glamorous and high-profile events.",

            # 펑크 스타일
            "This outfit represents punk fashion with bold patterns.",
            "This outfit embraces edgy and rebellious punk rock aesthetics.",
            "This outfit features leather, studs, and unconventional designs.",
            "This outfit reflects individuality and anti-mainstream vibes.",

            # 그라운드 룩 (캐주얼 비즈니스)
            "This outfit is smart and ideal for casual business wear.",
            "This outfit bridges formal and relaxed fashion styles.",
            "This outfit is suitable for semi-formal meetings or events.",
            "This outfit is polished yet approachable."

            # 코지 스타일
            "This outfit is cozy and perfect for cold weather.",
            "This outfit highlights warm fabrics like knitwear and fleece.",
            "This outfit is ideal for lounging comfortably at home.",
            "This outfit blends warmth and style for a cozy look."
        ]

        st.write("**Running object detection...**")
        image_cv2 = np.array(image)
        results = yolo_model.predict(source=np.array(image), imgsz=640, conf=0.25, device=device)
        detections = results[0].boxes
        detected_attributes = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls[0])]
            attributes = classify_attributes(resnet_model, image, (x1, y1, x2, y2), attribute_names, device)
            detected_attributes.extend(attributes)

            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        st.image(image_cv2, caption="Analyzed Image", use_column_width=True)

        if detected_attributes:
            st.write("**Detected Attributes:**")
            st.write(", ".join(set(detected_attributes)))

            matched_style = match_style(clip_model, image, detected_attributes, style_templates, device)
            st.write("**Predicted Styles:**")
            st.write(matched_style)

            is_match, suggestions = check_style_match(body, detected_attributes)
            if is_match:
                st.success("Great! The detected outfit matches your body type.")
            else:
                st.warning("This style might not suit your body type. Here are some suggestions:")
                st.write(", ".join(suggestions))
        else:
            st.write("No attributes detected.")


if __name__ == "__main__":
    main()

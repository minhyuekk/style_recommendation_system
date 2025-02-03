import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

model = YOLO('../YOLO/weights/deepfashion2_yolov8s-seg.pt')
test_source = '../img.jpg'

results = model.predict(source=test_source, imgsz=640, conf=0.25, device=0)

if __name__ == '__main__':
    result = results[0]
    img = cv2.imread(test_source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        font_scale = 0.5
        font_thickness = 1
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
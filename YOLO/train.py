from ultralytics import YOLO

model = YOLO('yolov8s.pt')
data_path = "../YOLO/data/dataset.yaml"

if __name__ == "__main__":
    model.train(data=data_path, epochs=100, imgsz=640, batch=16, device=0)
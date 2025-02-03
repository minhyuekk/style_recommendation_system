import os
import json

dataset = "../YOLO/data"
output = "../YOLO/data"

splits = ["train", "val", "test"]


def convert(box, img_width, img_height):
    x1, y1, w, h = box
    x_center = (x1 + w / 2) / img_width
    y_center = (y1 + h / 2) / img_height
    w = w / img_width
    h = h / img_height
    return x_center, y_center, w, h


if __name__ == "__main__":
    for split in splits:
        json_path = os.path.join(dataset, f"{split}.json")
        image_dir = os.path.join(dataset, split, "images")
        label_dir = os.path.join(output, "labels", split)
        os.makedirs(label_dir, exist_ok=True)

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        for img_id, data in annotations.items():
            img_width = data['image_width']
            img_height = data['image_height']
            labels = data['annotations']

            yolo_annotations = []
            for label in labels:
                class_id = label['category_id'] - 1
                bbox = label['bbox']
                yolo_bbox = convert(bbox, img_width, img_height)
                yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

            txt_path = os.path.join(label_dir, f"{img_id}.txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def mapping(label_file):
    """
    라벨 파일을 읽어 속성 ID 매핑과 이미지 ID-파일명 매핑을 생성.
    """
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {item["image_id"]: item["attribute_ids"] for item in data["annotations"]}

    id_to_filename = {item["id"]: item["file_name"] for item in data["images"]}

    unique_attribute_ids = set(attr_id for attrs in labels.values() for attr_id in attrs)
    attr_id_to_index = {attr_id: idx for idx, attr_id in enumerate(sorted(unique_attribute_ids))}

    return labels, id_to_filename, attr_id_to_index


class FashionpediaDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels, self.id_to_filename, self.attr_id_to_index = mapping(label_file)
        self.image_ids = list(self.labels.keys())
        self.transform = transform
        self.num_attributes = len(self.attr_id_to_index)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_ids = self.labels[image_id]
        label_indices = [self.attr_id_to_index[attr_id] for attr_id in label_ids]

        # 원-핫 인코딩
        label_tensor = torch.zeros(self.num_attributes)
        label_tensor[label_indices] = 1

        return image, label_tensor


def create_dataloader(image_dir, label_file, batch_size, shuffle=True):
    dataset = FashionpediaDataset(image_dir=image_dir, label_file=label_file, transform=resnet_preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    train_image_dir = "../feature_extractor/data/images/train"
    train_label_file = "../feature_extractor/data/labels/attributes_train2020.json"

    train_loader = create_dataloader(train_image_dir, train_label_file, batch_size=32)

    # 데이터 확인
    print("[*] 데이터 확인 시작")
    for images, labels in train_loader:
        print(f"이미지 배치 크기: {images.size()}")
        print(f"라벨 배치 크기: {labels.size()}")
        print(f"라벨 예시: {labels[0]}")
        break

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from datasets import create_dataloader
from train import ResNetAttributeModel

test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, num_attributes, device):
    model = ResNetAttributeModel(num_attributes=num_attributes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[*] Model loaded from {model_path}")
    return model


def calculate_accuracy(outputs, labels):
    preds = torch.sigmoid(outputs)
    preds = (preds > 0.5).float()
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return correct / total


def test_model(model, test_loader, device):
    model.eval()
    total_accuracy = 0.0

    print("[*] Testing model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)

            # Accuracy
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy.item()

    avg_accuracy = total_accuracy / len(test_loader)
    print(f"[*] Test Accuracy: {avg_accuracy:.4f}")


def sample_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    print(f"[*] Displaying {num_samples} sample predictions...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for j in range(min(num_samples, images.size(0))):
                print(f"\n[Sample {j+1}]")
                print(f"Ground Truth: {labels[j].cpu().numpy().astype(int)}")
                print(f"Prediction  : {preds[j].cpu().numpy().astype(int)}")

            num_samples -= images.size(0)
            if num_samples <= 0:
                break


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}")

    test_image_dir = "../feature_extractor/data/images/test"
    test_label_file = "../feature_extractor/data/labels/attributes_val2020.json"
    model_path = "../feature_extractor/weights/fashion_attribute_model.pth"
    num_attributes = 294

    print("[*] Preparing test data...")
    test_loader = create_dataloader(test_image_dir, test_label_file, batch_size=32, shuffle=False)

    model = load_model(model_path, num_attributes, device)

    test_model(model, test_loader, device)

    sample_predictions(model, test_loader, device, num_samples=5)

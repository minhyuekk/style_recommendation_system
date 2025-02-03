import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import models
from tqdm import tqdm

from datasets import create_dataloader


class ResNetAttributeModel(nn.Module):
    def __init__(self, num_attributes):
        super(ResNetAttributeModel, self).__init__()
        # ResNet50
        self.resnet = models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_attributes)

    def forward(self, x):
        return self.resnet(x)


def calculate_accuracy(outputs, labels):
    preds = torch.sigmoid(outputs)
    preds = (preds > 0.5).float()
    correct = (preds == labels).float().sum()
    total = labels.numel()
    accuracy = correct / total
    return accuracy


def train(model, train_loader, num_epochs, criterion, optimizer, device):
    model.train()
    print('[*] start training')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy.item()

            # wandb 로그
            wandb.log({"train_loss": loss.item(), "train_acc": accuracy.item()})

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        wandb.log({"epoch_loss": avg_loss, "epoch_accuracy": avg_acc})


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")


if __name__ == "__main__":
    wandb.init(project="fashion-attribute-classification", name="resnet50-training")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}")

    train_image_dir = "../feature_extractor/data/images/train"
    train_label_file = "../feature_extractor/data/labels/attributes_train2020.json"
    val_image_dir = "../feature_extractor/data/images/test"
    val_label_file = "../feature_extractor/data/labels/attributes_val2020.json"

    print('[*] preparing data')
    train_loader = create_dataloader(train_image_dir, train_label_file, batch_size=32)
    val_loader = create_dataloader(val_image_dir, val_label_file, batch_size=32, shuffle=False)

    num_attributes = 294
    learning_rate = 0.001
    num_epochs = 10

    model = ResNetAttributeModel(num_attributes=num_attributes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="all", log_freq=10)

    # Train
    train(model, train_loader, num_epochs, criterion, optimizer, device)

    # Evaluate
    # evaluate(model, val_loader, criterion, device)

    # Save
    torch.save(model.state_dict(), 'weights/fashion_attribute_model.pth')
    print("[*] Model saved")

    wandb.finish()

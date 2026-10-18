import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("data/train", transform=transform)
test_dataset = datasets.ImageFolder("data/test", transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0
)

images, labels = next(iter(train_loader))
print("Classes:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # two class scores (logits)

model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

if device.type == "cuda":
    torch.cuda.synchronize()
training_start = time.perf_counter()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Training accuracy: {accuracy:.2f}%"
    )

if device.type == "cuda":
    torch.cuda.synchronize()
training_time = time.perf_counter() - training_start
print(f"Training time: {training_time:.3f} seconds")

print("Output shape:", outputs.shape)

model.eval()
correct = 0
total = 0

if device.type == "cuda":
    torch.cuda.synchronize()
test_start = time.perf_counter()

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = model(test_images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == test_labels).sum().item()
        total += test_labels.size(0)

if device.type == "cuda":
    torch.cuda.synchronize()
test_time = time.perf_counter() - test_start

print(f"Test accuracy: {100.0 * correct / total:.2f}% ({correct}/{total})")
print(f"Test processing time: {test_time:.3f} seconds")
print(f"Average time per image: {test_time / total:.4f} seconds")

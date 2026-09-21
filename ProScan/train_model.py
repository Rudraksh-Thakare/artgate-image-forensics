import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# 🔥 Improved transform (IMPORTANT)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

DATASET_PATH = "Dataset"

# ================================
# 🔥 1. TRAIN CATEGORY DETECTOR
# ================================
print("\n===== TRAINING CATEGORY MODEL =====")

category_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
category_loader = DataLoader(category_dataset, batch_size=BATCH_SIZE, shuffle=True)

category_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
category_model.classifier[1] = nn.Linear(category_model.last_channel, len(category_dataset.classes))
category_model = category_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(category_model.parameters(), lr=0.001)

print("Categories:", category_dataset.classes)

for epoch in range(EPOCHS):
    total_loss = 0
    category_model.train()

    for images, labels in category_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = category_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Category Model | Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(category_model.state_dict(), "category_model.pth")
print("Saved: category_model.pth")


# ================================
# 🔥 2. TRAIN FAKE DETECTION MODELS
# ================================
print("\n===== TRAINING CATEGORY-WISE MODELS =====")

for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    print("\nTraining for category:", category)

    dataset = datasets.ImageFolder(category_path, transform=transform)

    if len(dataset) == 0:
        print("Skipping:", category)
        continue

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{category} | Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"{category.lower()}_model.pth")
    print(f"Saved: {category.lower()}_model.pth")
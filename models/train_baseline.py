import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

# -----------------------------------
# Device Configuration
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

# -----------------------------------
# Hyperparameters
# -----------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 2

# -----------------------------------
# Dataset Path
# -----------------------------------
DATA_DIR = "data/raw/PlantVillage"

# -----------------------------------
# Image Transformations
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------------
# Load Dataset
# -----------------------------------
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

# -----------------------------------
# Use Smaller Subset for Faster CPU Training
# -----------------------------------
subset_size = 10000

full_dataset = Subset(
    full_dataset,
    range(subset_size)
)

num_classes = 16

print(f"Number of Classes: {num_classes}")
print(f"Dataset Size Used: {subset_size}")

# -----------------------------------
# Dataset Split
# -----------------------------------
train_size = int(0.7 * subset_size)
val_size = int(0.15 * subset_size)
test_size = subset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

# -----------------------------------
# DataLoaders
# -----------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------------
# Load MobileNetV2
# -----------------------------------
model = models.mobilenet_v2(weights="DEFAULT")

# Replace Final Layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# -----------------------------------
# Loss Function and Optimizer
# -----------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------------
# Training Loop
# -----------------------------------
best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Accuracy Calculation
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # Progress Display
        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}]"
            )

    train_accuracy = 100 * correct / total

    # -----------------------------------
    # Validation
    # -----------------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Results")
    print(f"Train Loss: {running_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # -----------------------------------
    # Save Best Model
    # -----------------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "results/baseline_mobilenetv2.pth"
        )

        print("Best model saved!")

print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
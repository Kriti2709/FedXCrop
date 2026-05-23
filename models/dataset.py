import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Dataset path
DATA_DIR = "data/raw/PlantVillage"

# Image transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load dataset
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

# Dataset info
num_classes = len(full_dataset.classes)

print(f"Total Images: {len(full_dataset)}")
print(f"Number of Classes: {num_classes}")

print("\nClasses:")
for idx, class_name in enumerate(full_dataset.classes):
    print(f"{idx}: {class_name}")

# Split ratios
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset Split:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# Check one batch
images, labels = next(iter(train_loader))

print("\nBatch Shape:")
print(f"Images Shape: {images.shape}")
print(f"Labels Shape: {labels.shape}")
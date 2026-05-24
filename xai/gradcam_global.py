import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -----------------------------------
# Device
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

# -----------------------------------
# Constants
# -----------------------------------
IMAGE_SIZE = 128

IMAGE_PATH = "C:\\Users\\saini\\OneDrive\\Desktop\\FedXCrop\\data\\raw\\PlantVillage\\Potato___Early_blight\\0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"

MODEL_PATH = "results/baseline_mobilenetv2.pth"

NUM_CLASSES = 16

# -----------------------------------
# Load Model
# -----------------------------------
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")

# -----------------------------------
# Image Transform
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------------
# Load Image
# -----------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------------
# Prepare RGB Image for Visualization
# -----------------------------------
rgb_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

# -----------------------------------
# Target Layer
# -----------------------------------
target_layers = [model.features[-1]]

# -----------------------------------
# Create GradCAM Object
# -----------------------------------
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# -----------------------------------
# Generate CAM
# -----------------------------------
targets = [ClassifierOutputTarget(0)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

# -----------------------------------
# Overlay Heatmap
# -----------------------------------
visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

# -----------------------------------
# Plot Result
# -----------------------------------
plt.figure(figsize=(8, 8))

plt.imshow(visualization)

plt.title("Grad-CAM Explanation")

plt.axis("off")

# -----------------------------------
# Save Figure
# -----------------------------------
os.makedirs("results/figures", exist_ok=True)

save_path = "results/figures/gradcam_output.png"

plt.savefig(save_path)

print(f"\nGrad-CAM saved at:\n{save_path}")

plt.show()
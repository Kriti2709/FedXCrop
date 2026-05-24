import os
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# -----------------------------------
# Device
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

# -----------------------------------
# Constants
# -----------------------------------
IMAGE_SIZE = 128
NUM_CLASSES = 16

MODEL_PATH = "results/baseline_mobilenetv2.pth"

IMAGE_PATH = "data/raw/PlantVillage/Potato___Early_blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"

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

# Disable inplace ReLU6
for module in model.modules():

    if isinstance(module, nn.ReLU6):
        module.inplace = False

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

print("Input image loaded successfully!")

# -----------------------------------
# Background Dataset
# -----------------------------------
background = torch.randn(
    (10, 3, IMAGE_SIZE, IMAGE_SIZE)
).to(device)

print("Background dataset created!")

# -----------------------------------
# SHAP Explainer
# -----------------------------------
explainer = shap.GradientExplainer(
    model,
    background
)

print("SHAP GradientExplainer initialized!")

# -----------------------------------
# Generate SHAP Values
# -----------------------------------
shap_values = explainer.shap_values(
    input_tensor,
    nsamples=50
)

print("SHAP values generated successfully!")

# -----------------------------------
# Convert Input Image
# -----------------------------------
image_np = input_tensor.squeeze().cpu().numpy()

# CHW -> HWC
image_np = np.transpose(image_np, (1, 2, 0))

# Add batch dimension
image_np = np.expand_dims(image_np, axis=0)

# -----------------------------------
# Handle SHAP Dimensions
# -----------------------------------
# Expected weird shape:
# (1, 3, 128, 128, 16)

if isinstance(shap_values, list):

    shap_values = shap_values[0]

# Remove batch dimension
shap_values = shap_values[0]

# Select FIRST CLASS explanation
# Shape becomes (3,128,128)
shap_values = shap_values[:, :, :, 0]

# CHW -> HWC
shap_values = np.transpose(
    shap_values,
    (1, 2, 0)
)

# Add batch dimension
shap_values = np.expand_dims(
    shap_values,
    axis=0
)

print(f"Final SHAP Shape: {shap_values.shape}")
print(f"Final Image Shape: {image_np.shape}")

# -----------------------------------
# Create Results Directory
# -----------------------------------
os.makedirs("results/figures", exist_ok=True)

# -----------------------------------
# Plot SHAP
# -----------------------------------
shap.image_plot(
    shap_values,
    image_np,
    show=False
)

# -----------------------------------
# Save Figure
# -----------------------------------
save_path = "results/figures/shap_output.png"

plt.savefig(save_path)

print(f"\nSHAP visualization saved at:\n{save_path}")

plt.show()
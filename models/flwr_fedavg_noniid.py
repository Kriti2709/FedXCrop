import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# -----------------------------------
# Device
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

# -----------------------------------
# Hyperparameters
# -----------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_CLIENTS = 4
NUM_ROUNDS = 2

DIRICHLET_ALPHA = 0.5

# -----------------------------------
# Dataset
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    root="data/raw/PlantVillage",
    transform=transform
)

# Smaller subset for CPU execution
subset_size = 4000

dataset = Subset(dataset, range(subset_size))

num_classes = 16

# -----------------------------------
# Extract Labels
# -----------------------------------
targets = []

for i in range(len(dataset)):
    _, label = dataset[i]
    targets.append(label)

targets = np.array(targets)

# -----------------------------------
# Dirichlet Non-IID Split
# -----------------------------------
client_indices = [[] for _ in range(NUM_CLIENTS)]

for class_id in range(num_classes):

    class_indices = np.where(targets == class_id)[0]

    np.random.shuffle(class_indices)

    proportions = np.random.dirichlet(
        alpha=[DIRICHLET_ALPHA] * NUM_CLIENTS
    )

    proportions = (
        np.cumsum(proportions) * len(class_indices)
    ).astype(int)[:-1]

    split_indices = np.split(class_indices, proportions)

    for client_id in range(NUM_CLIENTS):

        client_indices[client_id].extend(
            split_indices[client_id]
        )

# -----------------------------------
# Create Client Datasets
# -----------------------------------
client_datasets = []

for client_id in range(NUM_CLIENTS):

    subset = Subset(dataset, client_indices[client_id])

    client_datasets.append(subset)

    print(
        f"Client {client_id} Dataset Size: "
        f"{len(subset)}"
    )

# -----------------------------------
# Model Definition
# -----------------------------------
def load_model():

    model = models.mobilenet_v2(weights="DEFAULT")

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model.to(device)

# -----------------------------------
# Train Function
# -----------------------------------
def train(model, loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# -----------------------------------
# Test Function
# -----------------------------------
def test(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# -----------------------------------
# Flower Client
# -----------------------------------
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader):

        self.model = load_model()

        self.trainloader = trainloader

    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = OrderedDict({
            k: torch.tensor(v)
            for k, v in params_dict
        })

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        accuracy = train(
            self.model,
            self.trainloader
        )

        print(f"Client Accuracy: {accuracy:.2f}%")

        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {"accuracy": accuracy}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        accuracy = test(
            self.model,
            self.trainloader
        )

        return 0.0, len(self.trainloader.dataset), {
            "accuracy": accuracy
        }

# -----------------------------------
# Client Function
# -----------------------------------
def client_fn(cid):

    trainloader = DataLoader(
        client_datasets[int(cid)],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    return FlowerClient(trainloader)

# -----------------------------------
# FedAvg Strategy
# -----------------------------------
strategy = fl.server.strategy.FedAvg()

# -----------------------------------
# Start Simulation
# -----------------------------------
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    ),
    strategy=strategy,
)
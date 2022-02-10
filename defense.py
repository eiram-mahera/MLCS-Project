import numpy as np
import pickle
import os
from tqdm import tqdm
from torchattacks import *

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.models as models
torch.manual_seed(17)
np.random.seed(200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "data/"
save_model = "checkpoint/"
performance = "performance/training-performance.pkl"
batch_size = 64
in_channels = 1
num_classes = 10
learning_rate = 0.001
num_epochs = 3


class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class SubModel2(nn.Module):
    def __init__(self):
        super(SubModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SubModel3(nn.Module):
    def __init__(self):
        super(SubModel3, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)

        return x


def train(train_loader, test_loader, model, file_name):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
    torch.save(model, os.path.join(save_model, file_name))

    # Check accuracy on training & test to see how good our model
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        model.train()
        return num_correct/num_samples

    train_accuracy = np.round(check_accuracy(train_loader, model) * 100, 2)
    test_accuracy = np.round(check_accuracy(test_loader, model) * 100, 2)
    return train_accuracy, test_accuracy


print("Loading data")
train_dataset = datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print("Training and testing all models")
results = {}
target_model = Target().to(device)
train_accuracy, test_accuracy = train(train_loader, test_loader, target_model, "target_model.pth")
results["Target Model"] = {"train": train_accuracy, "test": test_accuracy}

sub_model_1 = models.resnet50(pretrained=True)
sub_model_1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
sub_model_1.fc = nn.Linear(2048, 10, bias=True)
train_accuracy, test_accuracy = train(train_loader, test_loader, sub_model_1, "sub_model_1.pth")
results["Substitute Model 1"] = {"train": train_accuracy, "test": test_accuracy}

sub_model_2 = SubModel2().to(device)
train_accuracy, test_accuracy = train(train_loader, test_loader, sub_model_2, "sub_model_2.pth")
results["Substitute Model 2"] = {"train": train_accuracy, "test": test_accuracy}

sub_model_3 = SubModel3().to(device)
train_accuracy, test_accuracy = train(train_loader, test_loader, sub_model_3, "sub_model_3.pth")
results["Substitute Model 3"] = {"train": train_accuracy, "test": test_accuracy}

all_models = [target_model, sub_model_1, sub_model_2, sub_model_3]

attacks = {
    "FGSM": [
        FGSM(model, eps=8/255) for model in all_models
    ],
    "PGD": [
        PGD(model, eps=8/255, alpha=2/255, steps=4) for model in all_models
    ],
    "CW": [
        CW(model, c=10, lr=0.01, steps=10, kappa=10) for model in all_models
    ],
    "BIM": [
        BIM(model, eps=8/255, alpha=0.1, steps=4)
        for model in all_models
    ],
    "FGSM + PGD": [item for sublist in [
        (
            FGSM(model, eps=8/255),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + FGSM": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            FGSM(model, eps=8/255)
        ) for model in all_models
    ] for item in sublist],
    "FGSM + BIM": [item for sublist in [
        (
            FGSM(model, eps=8/255),
            BIM(model, eps=8/255, alpha=0.1, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + PGD": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "BIM + PGD": [item for sublist in [
        (
            BIM(model, eps=8/255, alpha=0.1, steps=4),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + BIM": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            BIM(model, eps=8/255, alpha=0.1, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + FGSM + PGD": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            FGSM(model, eps=8/255),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "FGSM + BIM + PGD": [item for sublist in [
        (
            FGSM(model, eps=8/255),
            BIM(model, eps=8/255, alpha=0.1, steps=4),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + FGSM + BIM": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            FGSM(model, eps=8/255),
            BIM(model, eps=8/255, alpha=0.1, steps=4),
        ) for model in all_models
    ] for item in sublist],
    "CW + BIM + PGD": [item for sublist in [
        (
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            BIM(model, eps=8/255, alpha=0.1, steps=4),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
    "CW + FGSM + BIM + PGD": [item for sublist in [
        (
            FGSM(model, eps=8/255),
            CW(model, c=10, lr=0.01, steps=10, kappa=10),
            PGD(model, eps=8/255, alpha=2/255, steps=4)
        ) for model in all_models
    ] for item in sublist],
}

train_loader_2 = DataLoader(dataset=train_dataset, batch_size=4000, shuffle=True)
adversarial_data = []

# use an ensemble of all the models and generate adversarial samples using cascading attacks
for (X, Y), (name, attack) in tqdm(zip(train_loader_2, attacks.items())):
    print(name)
    X_adv = X
    for atk in tqdm(attack):
        X_adv = atk(X_adv, Y)
    for x, y in zip(X_adv, Y):
        adversarial_data.append((x, y))

print(f"Generated {len(adversarial_data)} adversarial samples")
adv_train_loader = DataLoader(dataset=adversarial_data, batch_size=batch_size, shuffle=True)

# adversarial training
train_accuracy, test_accuracy = train(adv_train_loader, test_loader, target_model, "target_model_adv_trained.pth")

results["Adversarial Trained Target Model"] = {"train": train_accuracy, "test": test_accuracy}

print(results)

with open(performance, "wb") as fh:
    pickle.dump(results, fh)

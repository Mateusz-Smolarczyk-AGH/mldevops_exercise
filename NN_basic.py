import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import optuna

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 3,
    },
)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 28x28 image, Output: 128 features
        self.fc2 = nn.Linear(128, 64)  # Hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, 10)  # Output layer: 10 classes for classification

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image to a vector of size 28*28
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation at the output layer (will use CrossEntropyLoss which applies softmax)
        return x


def objective(trial):
    # Próbujemy różne wartości hiperparametrów
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_epochs = trial.suggest_int("num_epochs", 1, 10)

    # Aktualizujemy dane wejściowe z nowym batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Inicjalizujemy model i optymalizator z nowym learning_rate
    model = FashionMNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Trenujemy model z próbnymi hiperparametrami
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Testujemy model i obliczamy dokładność
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    # Logujemy wynik dla tej próby
    wandb.log(
        {"accuracy": accuracy, "learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs}
    )

    return accuracy


def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            running_loss += loss.item()
            wandb.log({"loss": loss})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


model = FashionMNISTModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)  # n_trials ustala liczbę prób do wykonania

train_model(model, train_loader, criterion, optimizer, num_epochs=8)


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")


# Evaluate the model
test_model(model, test_loader)

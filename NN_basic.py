import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import optuna
from typing import Any

# Log in to Weights and Biases (WandB)
wandb.login()

# Initialize WandB run with project name and initial hyperparameters
run = wandb.init(
    project="my-awesome-project",
    config={
        "learning_rate": 0.01,
        "epochs": 3,
    },
)

# Define transformations for the dataset: convert to tensor and normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load FashionMNIST dataset with specified transformations
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Initialize data loaders for training and testing datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FashionMNISTModel(nn.Module):
    """
    Neural network model for classifying FashionMNIST dataset images.

    This model consists of three fully connected (fc) layers:
    - fc1: Input layer (28x28 pixels, flattened), output 128 features
    - fc2: Hidden layer with 64 neurons
    - fc3: Output layer for 10 classes, one per item category

    Methods
    -------
    forward(x):
        Passes input through the network layers and returns logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer: 28x28 pixels, output: 128 features
        self.fc2 = nn.Linear(128, 64)  # Hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape [batch_size, 1, 28, 28].

        Returns
        -------
        torch.Tensor
            Logits for each class, shape [batch_size, 10].
        """
        x = x.view(-1, 28 * 28)  # Flatten the image to a 1D vector of 28*28 size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer without activation (CrossEntropyLoss will handle softmax)
        return x


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.

    This function sets up and trains a model with trial-defined hyperparameters, evaluates it on the test set,
    and returns its accuracy for Optuna to optimize.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object that suggests values for hyperparameters.

    Returns
    -------
    float
        Accuracy of the model on the test dataset.
    """
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_epochs = trial.suggest_int("num_epochs", 1, 10)

    # Update data loaders with the new batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer with trial's learning rate
    model = FashionMNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model with the sampled hyperparameters
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate model accuracy on test set
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

    # Log results to WandB
    wandb.log(
        {"accuracy": accuracy, "learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs}
    )

    return accuracy  # Return accuracy as the objective score


def train_model(
    model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 1
) -> None:
    """
    Trains the neural network model for a specified number of epochs.

    Parameters
    ----------
    model : nn.Module
        Neural network model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    criterion : nn.Module
        Loss function to optimize.
    optimizer : optim.Optimizer
        Optimizer for model parameters.
    num_epochs : int, optional
        Number of epochs to train the model (default is 1).
    """
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


# Run Optuna hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)  # Number of trials to perform

# Print best hyperparameters and accuracy
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)


def test_model(model: nn.Module, test_loader: DataLoader) -> None:
    """
    Evaluates the model on the test set and prints accuracy.

    Parameters
    ----------
    model : nn.Module
        Trained neural network model to evaluate.
    test_loader : DataLoader
        DataLoader for the test dataset.
    """
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


# Initialize and train the final model with the best hyperparameters
model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, num_epochs=8)

# Evaluate the model
test_model(model, test_loader)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()  # Set the model to training mode
    running_loss = 0.0

    epoch_correct = 0
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=(len(train_loader))):
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        epoch_correct += num_correct(outputs, labels)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print training loss for this epoch
    print(
        f"Training Loss: {running_loss / len(train_loader)}, Avg. Accuracy: {epoch_correct / len(train_loader.dataset)}"
    )


def validate(model: nn.Module, dataloader: DataLoader):
    # Evaluate the model on the test data
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=(len(dataloader))):
            outputs = model(inputs)
            total += labels.size(0)
            correct += num_correct(outputs, labels)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    return accuracy


def num_correct(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item()

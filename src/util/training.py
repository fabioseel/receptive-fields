import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.regularization import ActivationRegularization, WeightRegularization


def train(model: nn.Module, optimizer: optim.Optimizer, act_regularizer:ActivationRegularization, weight_regularizer:WeightRegularization, train_loader: DataLoader, device: torch.device, max_num_batches: int = None):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()  # Set the model to training mode
    running_loss = 0.0

    epoch_correct = 0
    num_batches = len(train_loader)
    if max_num_batches is not None:
        num_batches = min(num_batches, max_num_batches)
    pbar = tqdm(enumerate(train_loader), total=num_batches)
    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        batch_correct =  num_correct(outputs, labels)
        epoch_correct += batch_correct
        loss = criterion(outputs, labels) + act_regularizer.penalty() + weight_regularizer.penalty()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix({'Acc.:': batch_correct/len(labels), 'Loss': loss.item()})
    return running_loss/num_batches, epoch_correct/min(num_batches*train_loader.batch_size, len(train_loader.dataset))


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    # Evaluate the model on the test data
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=(len(dataloader))):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            correct += num_correct(outputs, labels)
    accuracy = 100 * correct / total
    return accuracy


def num_correct(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_builder import load_model
from training import train, validate
import torch

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)

args = parser.parse_args()

filepath = args.config

model = load_model(filepath)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-6)


transf = [transforms.ToTensor()]
if model.in_channels == 1:
    transf.append(transforms.Grayscale())
train_data = datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transforms.Compose(transf)
)
test_data = datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transforms.Compose(transf)
)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, prefetch_factor=4)

prev_best_acc = 0
early_stop_epochs = 5
early_stop = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
model.to(device)

i = 0
while not early_stop:
    epoch_train_loss, epoch_train_acc = train(model, optimizer, train_loader, device)
    # Print training loss for this epoch
    print(
        f"Epoch {i} - Training Loss: {epoch_train_loss}, Avg. Accuracy: {epoch_train_acc}"
    )
    epoch_val_acc = validate(model, test_loader, device)
    print(f"Epoch {i} - Test Accuracy: {epoch_val_acc}%")
    if epoch_val_acc > prev_best_acc:
        prev_best_acc = epoch_val_acc
        inc_count = 0
        model.save(filepath)
    else:
        inc_count += 1
        if inc_count > early_stop_epochs:
            early_stop = True
    i+=1
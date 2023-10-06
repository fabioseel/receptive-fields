import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple import SimpleCNN
from models.lindsey import LindseyNet
from training import train, validate

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("batch_size", type=int)

args = parser.parse_args()

filepath = Path(args.config)

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config['type'] == "lindsey":
    model = LindseyNet(**config['config'])
else:
    model = SimpleCNN(**config['config'])
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_data = datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

prev_best_acc = 0
early_stop_epochs = 5
early_stop = False

while not early_stop:
    train(model, optimizer, train_loader)
    accuracy = validate(model, test_loader)
    if accuracy > prev_best_acc:
        prev_best_acc = accuracy
        inc_count = 0
        model.save(filepath.with_suffix("").as_posix())
    else:
        inc_count += 1
        if inc_count > early_stop_epochs:
            early_stop = True

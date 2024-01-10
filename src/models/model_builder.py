from receptive_fields.models.designed import DesignedModel
from receptive_fields.models.lindsey import LindseyNet
from receptive_fields.models.simple import SimpleCNN
from receptive_fields.models.retinal import RetinalModel
from receptive_fields.models.alexnet import AlexNet
from receptive_fields.models.base_model import BaseModel
from torch import nn
import yaml

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def load_model(path, weights_file=None) -> BaseModel:
    with open(path+".cfg", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config['type'] == "lindsey":
        model = LindseyNet.load(path, weights_file)
    elif config['type'] == "retinal":
        model = RetinalModel.load(path, weights_file)
    elif config['type'] == "designed":
        model = DesignedModel.load(path, weights_file)
    elif config['type'] == "alexnet":
        model = AlexNet.load(path, weights_file)
    else:
        model = SimpleCNN.load(path, weights_file)
    return model

def open_experiment(path, train_data=True, test_data=False, batch_size=10):
    model = load_model(path)
    transf = [transforms.ToTensor()]
    in_channels = model.in_channels
    img_size = model.img_size
    if in_channels==1:
        transf.append(transforms.Grayscale())
    if img_size != 32:
        transf.append(transforms.Resize(img_size, antialias=True))

    if train_data==True:
        train_data = datasets.CIFAR10(root="../data", train=True, download=True, transform=transforms.Compose(transf))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    if test_data== True:
        test_data = datasets.CIFAR10(root="../data", train=False, download=True, transform=transforms.Compose(transf))
        test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    if train_data and test_data:
        return model, train_loader, test_loader
    elif train_data:
        return model, train_loader
    elif test_data:
        return model, test_loader
    else:
        return model
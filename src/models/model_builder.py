from models.designed import DesignedModel
from models.lindsey import LindseyNet
from models.simple import SimpleCNN
from models.retinal import RetinalModel
from torch import nn
import yaml

def load_model(path) -> nn.Module:
    with open(path+".cfg", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config['type'] == "lindsey":
        model = LindseyNet.load(path)
    elif config['type'] == "retinal":
        model = RetinalModel.load(path)
    elif config['type'] == "designed":
        model = DesignedModel.load(path)
    else:
        model = SimpleCNN.load(path)
    return model
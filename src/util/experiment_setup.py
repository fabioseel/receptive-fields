from receptive_fields.models.designed import DesignedModel
from receptive_fields.models.lindsey import LindseyNet
from receptive_fields.models.simple import SimpleCNN
from receptive_fields.models.retinal import RetinalModel
from receptive_fields.models.alexnet import AlexNet
from receptive_fields.models.base_model import BaseModel
from receptive_fields.util.dataset_transforms import RandomResize, ComposeImage
from torch import nn
import torch
import PIL
import yaml
import os.path as osp
import argparse

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

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("--optim", type=str, default="rmsprop")
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--weight_regularize", type=float, default=0)
    parser.add_argument("--weight_norm", type=float, default=2)
    parser.add_argument("--act_regularize", type=float, default=0)
    parser.add_argument("--act_norm", type=float, default=2)
    parser.add_argument("--save_hist", action="store_true", 
                        help="save the weights after each epoch") 
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="stop after n_epochs epoch") 
    parser.add_argument("--early_stop", type=int, default=0, 
                        help="stop after n epochs of not improving") 
    parser.add_argument("--max_num_batches", type=int, default=None, 
                        help="use only n batches during training")
    parser.add_argument("--enable_img_transforms", action="store_true", 
                        help="enable the image transforms that can be defined with add background and min/max resize")
    parser.add_argument("--add_background", type=str, default="rl", 
                        help="add the rl or black background and place the image at a random position")
    parser.add_argument("--min_resize", type=float, default=1,
                        help="the minimum factor the image is resized by before being composed onto the background") 
    parser.add_argument("--max_resize", type=float, default=1,
                        help="the maximum factor the image is resized to before being composed onto the background") 
    return parser

def get_parser_defaults():
    parser = create_parser()
    return vars(parser.parse_args(["","",0,0]))

def load_log(path):
    parser_def = get_parser_defaults()
    if osp.exists(path):
        with open(path, "r") as f: 
            log = yaml.load(f, Loader=yaml.FullLoader)
        for key in log.keys():
            parser_def[key] = log[key]
    return parser_def

def open_experiment(path, train_data=True, test_data=False, batch_size=10):
    model = load_model(path)
    log_path = osp.join(osp.split(path)[0], "logs",osp.split(path)[1]+".yaml")
    log = load_log(log_path)
    dataset_s = load_dataset(log["dataset"], train_data, test_data, log["enable_img_transforms"], log["min_resize"], log["max_resize"], log["add_background"], model.in_channels==1, model.img_size)

    if train_data and test_data:
        return model, *dataset_s
    elif train_data or test_data:
        return model, dataset_s
    else:
        return model
    
def setup_dataset_transforms(enable_img_transforms: bool, min_resize: float, max_resize: float, add_background: str="rl", grayscale: bool=False, img_size:int = None):
    transf = []
    if enable_img_transforms:
        transf.append(RandomResize((min_resize, max_resize)))
        if add_background == "rl":
            bg = PIL.Image.open("../resources/empty-viewport.png").convert("RGB")
            bg = bg.resize(img_size)
        else:
            bg = torch.zeros((3, *img_size[::-1]))
        transf.append(ComposeImage(bg))

    transf.append(transforms.ToTensor())
    if grayscale:
        transf.append(transforms.Grayscale())

    if not enable_img_transforms and img_size is not None:
            transf.append(transforms.Resize(img_size, antialias=True))
    return transforms.Compose(transf)

def load_dataset(dataset:str = "cifar10", train_data:bool = True, test_data:bool = False, enable_img_transforms: bool = False, min_resize: float = 1, max_resize: float = 1, add_background: str = "rl", grayscale: bool = False, img_size:int = None, data_root = "../data"):
    assert train_data or test_data

    if dataset == "stl10":
        transf = setup_dataset_transforms(enable_img_transforms, min_resize, max_resize, add_background, grayscale, img_size if img_size != 96 else None)
        if train_data:
            train_set = datasets.STL10(root=data_root, split="train", download=True, transform=transf)
        if test_data:
            test_set = datasets.STL10(root=data_root, split="test", download=True, transform=transf)
    else:
        transf = setup_dataset_transforms(enable_img_transforms, min_resize, max_resize, add_background, grayscale, img_size if img_size != 32 else None)
        if train_data:
            train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transf)
        if test_data:
            test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transf)
    if train_data and test_data:
        return train_set, test_set
    elif train_data:
        return train_set
    elif test_data:
        return test_set
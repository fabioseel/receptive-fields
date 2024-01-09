import math
import os
from torch.nn.modules.utils import _pair

import torch
import torch.nn as nn
import yaml
from abc import ABC, abstractmethod

from util.files import find_files_in_folder


class BaseModel(nn.Module, ABC):
    def __init__(self, img_size, activation="relu") -> None:
        super().__init__()
        self.img_size = img_size
        self.activation = activation

        if self.activation == "elu":
            self._activation_func = nn.ELU(inplace=True)
        elif self.activation == "selu":
            self._activation_func = nn.SELU(inplace=True)
        elif self.activation == "gelu":
            self._activation_func = nn.GELU()
        elif self.activation == "tanh":
            self._activation_func = nn.Tanh()
        elif self.activation == "leaky":
            self._activation_func = nn.LeakyReLU(negative_slope=0.2)
        else: # relu or anything else
            self._activation_func = nn.ReLU(inplace=True)

    @abstractmethod
    def get_sequential(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def _config(self) -> dict:
        pass

    @property
    def config(self) -> dict:
        conf = self._config
        conf["img_size"]= self.img_size
        conf["activation"]= self.activation
        return {
            "type": self.classname,
            "config": conf
        }

    @property
    def classname(self) -> str:
        return self.__class__

    def save(self, filename, save_cfg=True):
        config = self.config
        if save_cfg:
            with open(filename + ".cfg", "w") as f:
                yaml.dump(config, f)
        torch.save(self.state_dict(), filename + ".pth")

    @classmethod
    def load(cls, model_path, weights_file=None):
        with open(model_path + ".cfg", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        model = cls(**config['config'])
            
        if weights_file is None:
            if os.path.isdir(model_path):
                weights_files = find_files_in_folder(model_path, ".pth")
                weights_file = weights_files[-1]
            else:
                weights_file = model_path + ".pth"
        if os.path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file))
            except:
                model.load_state_dict(torch.load(weights_file, map_location=torch.device("cpu")))
        return model

    @property
    def img_size(self) -> [int, int]:
        return self._img_size
    @img_size.setter
    def img_size(self, value):
        self._img_size = _pair(value)
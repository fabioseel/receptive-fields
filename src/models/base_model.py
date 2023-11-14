import math
from os import path
from torch.nn.modules.utils import _pair

import torch
import torch.nn as nn
import yaml
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.img_size = img_size

    @abstractmethod
    def get_sequential(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        pass

    def save(self, filename, save_cfg=True):
        config = self.config()
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
            weights_file = model_path + ".pth"
        if path.exists(weights_file):
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
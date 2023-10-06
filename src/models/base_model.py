import math
from os import path

import torch
import torch.nn as nn
import yaml
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def get_sequential(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        pass

    def save(self, filename):
        config = self.config()
        with open(filename + ".cfg", "w") as f:
            yaml.dump(config, f)
        torch.save(self.state_dict(), filename + ".pth")

    @classmethod
    def load(cls, filename):
        with open(filename + ".cfg", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        model = cls(**config['config'])
        weights_file = filename + ".pth"
        if path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file))
            except:
                model.load_state_dict(torch.load(weights_file, map_location=torch.device("cpu")))
        return model

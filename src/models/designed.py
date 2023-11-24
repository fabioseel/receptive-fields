import torch.nn as nn
from models.base_model import BaseModel
from torch import nn
import torch
from modules import L2Pool, SeparableConv2d

class DesignedModel(BaseModel):
    def __init__(self, 
        img_size,
        num_classes,
        in_channels=3,
        n_base_channels = 16,
        n_lgn_channels = 32,
        n_fully_connected = 128):
        super(DesignedModel, self).__init__(img_size)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.n_base_channels = n_base_channels
        self.n_lgn_channels = n_lgn_channels
        self.n_fully_connected = n_fully_connected


        self.retina = nn.Sequential()
        self.fc = nn.Sequential()

        # BP
        self.retina.append(nn.Conv2d(in_channels, in_channels, kernel_size=11, groups=in_channels)) # A big kernel is (parameterwise) "cheap" at the first layer, since there are only 3 in_channels
        self.retina.append(nn.Conv2d(in_channels, n_base_channels, kernel_size=1))
        self.retina.append(self._activation_func)
        self.retina.append(L2Pool(kernel_size=3))

        # RGC
        self.retina.append(nn.Conv2d(n_base_channels, 2*n_base_channels, kernel_size=3, groups=n_base_channels))
        self.retina.append(nn.Conv2d(2*n_base_channels, 2*n_base_channels, kernel_size=1))
        self.retina.append(self._activation_func)
        self.retina.append(L2Pool(kernel_size=3)) # potentially replace Pool with relative large kernel (but strided), also compress number of channels

        # LGN
        self.retina.append(nn.Conv2d(2*n_base_channels, n_lgn_channels, kernel_size=1))
        self.retina.append(self._activation_func)

        # V1
        self.retina.append(nn.Conv2d(n_lgn_channels, 4*n_base_channels, kernel_size=4, groups=n_lgn_channels))
        self.retina.append(nn.Conv2d(4*n_base_channels, 4*n_base_channels, kernel_size=1))
        self.retina.append(self._activation_func)
        self.retina.append(L2Pool(kernel_size=4))                                                # potentially use Adaptive Avg Pool to make sure Img is not too big
        self.retina.append(nn.Flatten())

        # FC Layers
        x=torch.empty((1,in_channels, img_size, img_size))
        test_out = self.retina(x)
        self.fc.append(nn.Linear(in_features=test_out.shape[1], out_features=n_fully_connected))
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=n_fully_connected))
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=num_classes))

    @property
    def classname(self) -> str:
        return "designed"
    
    @property
    def _config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "n_base_channels": self.n_base_channels,
            "n_lgn_channels": self.n_lgn_channels,
            "n_fully_connected" : self.n_fully_connected
        }
    
    def get_sequential(self) -> nn.Module:
        seq = nn.Sequential()
        seq.extend(self.retina)
        seq.extend(self.fc)
        seq.append(nn.Softmax(dim=-1))
        return seq

    def forward(self, x):
        res = self.fc(self.retina(x))
        if not self.training:
            res = nn.functional.softmax(res, dim=-1)
        return res
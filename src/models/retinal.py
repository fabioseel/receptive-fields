import torch.nn as nn
from models.base_model import BaseModel
from torch import nn
import torch
from modules import GaborConv2d

class RetinalModel(BaseModel):
    def __init__(self, 
        img_size,
        num_classes,
        in_channels=3,
        n_base_channels=16,
        n_lgn_channels=32,
        n_fully_connected = 128,
        padding = None,
        ceil_mode=False,
        l1_conv_type="regular",
        l1_kernel_size=3):
        super(RetinalModel, self).__init__(img_size)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.n_base_channels = n_base_channels
        self.n_lgn_channels = n_lgn_channels
        self.n_fully_connected = n_fully_connected
        self.l1_conv_type = l1_conv_type
        self.l1_kernel_size = l1_kernel_size

        self.retina = nn.Sequential()
        self.fc = nn.Sequential()

        padding = 0 if padding is None else padding
        # BP
        if l1_conv_type == "gabor":
            self.retina.append(GaborConv2d(in_channels, n_base_channels, kernel_size=l1_kernel_size, padding=padding))
        else: # "regular" or anything else not specified
            self.retina.append(nn.Conv2d(in_channels, n_base_channels, kernel_size=l1_kernel_size, padding=padding))
        self.retina.append(nn.ELU())
        self.retina.append(nn.AvgPool2d(kernel_size=3, padding=padding, ceil_mode=ceil_mode))

        # RGC
        self.retina.append(nn.Conv2d(n_base_channels, 2*n_base_channels,kernel_size=3, padding=padding))
        self.retina.append(nn.ELU())
        self.retina.append(nn.AvgPool2d(kernel_size=3, padding=padding, ceil_mode=ceil_mode))

        # LGN
        self.retina.append(nn.Conv2d(2*n_base_channels, n_lgn_channels, kernel_size=1))
        self.retina.append(nn.ELU())

        # V1
        self.retina.append(nn.Conv2d(n_lgn_channels, 4*n_base_channels, kernel_size=4, padding=padding))
        self.retina.append(nn.ELU())
        self.retina.append(nn.MaxPool2d(kernel_size=4, padding=padding, ceil_mode=ceil_mode))
        self.retina.append(nn.Flatten())

        # FC Layers
        x=torch.empty((1,in_channels, self.img_size[0], self.img_size[1]))
        test_out = self.retina(x)
        self.fc.append(nn.Linear(in_features=test_out.shape[1], out_features=n_fully_connected))
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=n_fully_connected))
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=num_classes))

    def config(self) -> dict:
        return {
            "type" : "retinal",
            "config" : {
            "img_size": self.img_size,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "n_base_channels": self.n_base_channels,
            "n_lgn_channels": self.n_lgn_channels,
            "n_fully_connected": self.n_fully_connected,
            "l1_conv_type": self.l1_conv_type,
            "l1_kernel_size": self.l1_kernel_size
        }}
    
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
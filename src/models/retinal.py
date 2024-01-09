import torch.nn as nn
from models.base_model import BaseModel
from torch import nn
import torch
from util.modules import GaborConv2d, SpaceToDepth

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
        l1_kernel_size=3,
        l1_n_channels=None,
        lgn_kernel_size = 1,
        rgc_kernel_size = 3,
        v1_kernel_size = 4,
        pool = [True, True, True],
        activation="elu",
        fc_act=False, # forgot to put act for fc layers in the beginning, should be set to True always!
        stride= [1,1,1,1],
        spd=[1, 1, 1]):
        super(RetinalModel, self).__init__(img_size, activation)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.n_base_channels = n_base_channels
        self.n_lgn_channels = n_lgn_channels
        self.n_fully_connected = n_fully_connected
        self.l1_conv_type = l1_conv_type
        self.l1_kernel_size = l1_kernel_size
        self.l1_n_channels = n_base_channels if l1_n_channels is None else l1_n_channels
        self.lgn_kernel_size = lgn_kernel_size
        self.rgc_kernel_size = rgc_kernel_size
        self.v1_kernel_size = v1_kernel_size
        self.pool = pool
        self.spd = spd
        self.fc_act = fc_act

        self.retina = nn.Sequential()
        self.fc = nn.Sequential()

        padding = 0 if padding is None else padding
        # BP
        if self.spd[0]>1:
            self.retina.append(SpaceToDepth(factor=self.spd[0]))
        conv_in_channels = in_channels*self.spd[0]**2
        if l1_conv_type == "gabor":
            self.retina.append(GaborConv2d(conv_in_channels, self.l1_n_channels, kernel_size=l1_kernel_size, padding=padding))
        else: # "regular" or anything else not specified
            self.retina.append(nn.Conv2d(conv_in_channels, self.l1_n_channels, kernel_size=l1_kernel_size, padding=padding))
        self.retina.append(self._activation_func)
        if(self.pool[0]):
            self.retina.append(nn.AvgPool2d(kernel_size=3, padding=padding, ceil_mode=ceil_mode))

        # RGC
        if self.spd[1]>1:
            self.retina.append(SpaceToDepth(factor=self.spd[1]))
        self.retina.append(nn.Conv2d(self.l1_n_channels*self.spd[1]**2, 2*n_base_channels,kernel_size=self.rgc_kernel_size, padding=padding))
        self.retina.append(self._activation_func)
        if(self.pool[1]):
            self.retina.append(nn.AvgPool2d(kernel_size=3, padding=padding, ceil_mode=ceil_mode))

        # LGN
        self.retina.append(nn.Conv2d(2*n_base_channels, n_lgn_channels, kernel_size=self.lgn_kernel_size))
        self.retina.append(self._activation_func)

        # V1
        if self.spd[2]>1:
            self.retina.append(SpaceToDepth(factor=self.spd[2]))
        self.retina.append(nn.Conv2d(n_lgn_channels*self.spd[2]**2, 4*n_base_channels, kernel_size=self.v1_kernel_size, padding=padding))
        self.retina.append(self._activation_func)

        if(self.pool[0]):
            self.retina.append(nn.MaxPool2d(kernel_size=4, padding=padding, ceil_mode=ceil_mode))
        self.retina.append(nn.Flatten())

        # FC Layers
        x=torch.empty((1,in_channels, self.img_size[0], self.img_size[1]))
        test_out = self.retina(x)
        self.fc.append(nn.Linear(in_features=test_out.shape[1], out_features=n_fully_connected))
        if self.fc_act:
            self.fc.append(self._activation_func)
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=n_fully_connected))
        if self.fc_act:
            self.fc.append(self._activation_func)
        self.fc.append(nn.Linear(in_features=n_fully_connected, out_features=num_classes))

    @property
    def classname(self) -> str:
        return "retinal"
    
    @property
    def _config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "n_base_channels": self.n_base_channels,
            "n_lgn_channels": self.n_lgn_channels,
            "n_fully_connected": self.n_fully_connected,
            "l1_conv_type": self.l1_conv_type,
            "l1_kernel_size": self.l1_kernel_size,
            "l1_n_channels": self.l1_n_channels,
            "lgn_kernel_size": self.lgn_kernel_size,
            "rgc_kernel_size": self.rgc_kernel_size,
            "v1_kernel_size": self.v1_kernel_size,
            "pool": self.pool,
            "spd": self.spd,
            "fc_act": self.fc_act
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
# Reimplementation of the model used by Lindsey et al (2019), keras implementation: https://github.com/ganguli-lab/RetinalResources/blob/master/TrainModel.py

from torch import nn
import torch
import yaml
from os import path
import math
from models.base_model import BaseModel
from modules import space_to_depth

def calc_same_pad(i: int, k: int, s: int=1, d: int = 1) -> int:
        return max(math.ceil(((s - 1) * i - s + k) / 2), 0)

class LindseyNet(BaseModel):
    def __init__(
        self,
        img_size,
        num_classes,
        in_channels=3,
        kernel_size=9,
        retina_layers=2,
        retina_channels=32,
        retina_out_stride=1,
        bottleneck_channels=1,
        vvs_layers=2,
        vvs_channels = 32,
        first_fc = 1024,
        activation = "relu",
        dropout = 0.0,
        pool_retina = False,
        pool_vvs = False,
        spd_entry = False
    ):
        super(LindseyNet, self).__init__(img_size, activation)
        
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.retina_layers=retina_layers
        self.retina_channels=retina_channels
        self.retina_out_stride=retina_out_stride
        self.bottleneck_channels=bottleneck_channels
        self.vvs_layers=vvs_layers
        self.vvs_channels =vvs_channels
        self.first_fc = first_fc
        self.dropout = dropout
        self.pool_retina = pool_retina
        self.pool_vvs = pool_vvs
        self.spd_entry = spd_entry

        # Define Retina
        self.retina = nn.Sequential()
        if spd_entry:
            self.retina.append(space_to_depth())        
        for layer in range(retina_layers):
            _in_channels = retina_channels
            if layer == 0:
                _in_channels = in_channels if not spd_entry else in_channels * 4

            _out_channels = retina_channels
            _stride = 1    
            if layer == retina_layers - 1:
                _out_channels = bottleneck_channels
                _stride =  retina_out_stride

            _padding = calc_same_pad(self.img_size[0]//(1+self.spd_entry), kernel_size, _stride)
            self.retina.append(nn.Conv2d(_in_channels, _out_channels, kernel_size=kernel_size, stride=_stride, padding=_padding))
            self.retina.append(self._activation_func)
            if self.dropout > 0:
                self.retina.append(nn.Dropout(self.dropout))
            if self.pool_retina:
                self.retina.append(nn.AvgPool2d(2))

        # Define VVS
        self.vvs = nn.Sequential()
        for layer in range(vvs_layers):
            _in_channels = vvs_channels if layer != 0 else bottleneck_channels

            _padding = calc_same_pad(self.img_size[0], kernel_size)
            self.vvs.append(nn.Conv2d(_in_channels, vvs_channels, kernel_size=kernel_size, padding=_padding))
            self.vvs.append(self._activation_func)
            if self.dropout > 0:
                self.vvs.append(nn.Dropout(self.dropout))
            if self.pool_vvs:
                self.vvs.append(nn.AvgPool2d(2))

        self.fc = nn.Sequential()
        self.fc.append(nn.Flatten())

        x=torch.empty((1,in_channels, self.img_size[0], self.img_size[1]))
        test_out = self.retina(x)
        test_out = self.vvs(test_out)
        test_out = self.fc(test_out)
        self.fc.append(nn.Linear(test_out.shape[1], first_fc))
        self.fc.append(self._activation_func)
        self.fc.append(nn.Linear(first_fc, num_classes))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # TODO: Add noise potentially
        retina_out = self.retina(x)
        x = self.vvs(retina_out)
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x
    
    def get_sequential(self):
        seq = nn.Sequential()
        seq.extend(self.retina)
        seq.extend(self.vvs)
        seq.extend(self.fc)
        seq.append(self.softmax) # TODO: Proper handling with the softmax...
        return seq
    
    @property
    def classname(self) -> str:
        return "lindsey"
    
    @property
    def _config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "kernel_size": self.kernel_size,
            "retina_layers": self.retina_layers,
            "retina_channels":self.retina_channels,
            "retina_out_stride":self.retina_out_stride,
            "bottleneck_channels":self.bottleneck_channels,
            "vvs_layers":self.vvs_layers,
            "vvs_channels" :self.vvs_channels,
            "first_fc" : self.first_fc,
            "dropout" : self.dropout,
            "pool_retina" : self.pool_retina,
            "pool_vvs" : self.pool_vvs,
            "spd_entry" : self.spd_entry
        }

import math

import torch.nn as nn
import torch

from receptive_fields.models.base_model import BaseModel
from receptive_fields.util.modules import get_convolution
from torch.nn.modules.utils import _pair
from receptive_fields.util.modules import SpaceToDepth

class SimpleCNN(BaseModel):
    def __init__(
        self,
        img_size,
        num_classes,
        num_layers=3,
        num_fc_layers = 1,
        fc_dim = 128,
        in_channels=3,
        num_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        separable=False,
        num_skip_layers=None,
        gabor=False,
        staggered = False,
        activation="relu",
        pooling_ks = 1,
        spd = 1,
        pad_spd=True
    ):
        super(SimpleCNN, self).__init__(img_size, activation)
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_fc_layers = num_fc_layers
        self.fc_dim = fc_dim
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.separable = separable
        self.num_skip_layers=num_skip_layers
        self.gabor=gabor
        self.staggered = staggered
        self.pooling_ks = pooling_ks
        self.spd = spd
        self.pad_spd = pad_spd

        assert not(gabor and separable and staggered)

        self.space_to_depth = SpaceToDepth(factor=self.spd, pad=self.pad_spd)
        self.pool = nn.AvgPool2d(kernel_size=self.pooling_ks)

        # Define the first convolutional layer
        self.conv1 = get_convolution(
            in_channels*self.spd**2, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers, gabor, self.staggered
        )
        self.softmax = nn.Softmax(dim=-1)

        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.extra_conv_layers.append(
                get_convolution(
                    num_channels*self.spd**2, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers, gabor, self.staggered
                )
            )

        # Fully connected layer
        x=torch.empty((1,in_channels, self.img_size[0], self.img_size[1]))
        res_size = self.forward_conv(x).shape
        fc_in = torch.prod(torch.tensor(res_size))
        if self.num_fc_layers > 1:
            self.fc = nn.Sequential()
            self.fc.append(nn.Linear(fc_in, fc_dim))
            self.fc.append(self._activation_func)
            for i in range(self.num_fc_layers-1):
                self.fc.append(nn.Linear(self.fc_dim, self.fc_dim))
                self.fc.append(self._activation_func)
            self.fc.append(nn.Linear(self.fc_dim, num_classes))
        else:
            self.fc = nn.Linear(fc_in, num_classes)

    def forward_conv(self, x):
        if self.spd !=1:
            x = self.space_to_depth(x)
        x = self.conv1(x)
        x = self._activation_func(x)
        if self.pooling_ks !=1:
            x = self.pool(x)

        for conv_layer in self.extra_conv_layers:
            if self.spd !=1:
                x = self.space_to_depth(x)
            x = conv_layer(x)
            x = self._activation_func(x)
            if self.pooling_ks !=1:
                x = self.pool(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x

    def get_sequential(self): # TODO: how to add skip connections here?
        seq = nn.Sequential()

        if self.spd !=1:
            seq.append(self.space_to_depth)
        seq.append(self.conv1)
        seq.append(self._activation_func)
        if self.pooling_ks !=1:
            seq.append(self.pool)

        for conv_layer in self.extra_conv_layers:
            if self.spd !=1:
                seq.append(self.space_to_depth)
            seq.append(conv_layer)
            seq.append(self._activation_func)
            if self.pooling_ks !=1:
                seq.append(self.pool)
        seq.append(nn.Flatten())
        if self.num_fc_layers > 1:
            seq.extend(self.fc)
        else:
            seq.append(self.fc)
        seq.append(self.softmax)
        return seq

    @property
    def classname(self) -> str:
        return "simple"
    
    @property
    def _config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "num_fc_layers": self.num_fc_layers,
            "fc_dim": self.fc_dim,
            "in_channels": self.in_channels,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "separable": self.separable,
            "num_skip_layers": self.num_skip_layers,
            "gabor": self.gabor,
            "staggered": self.staggered,
            "pooling_ks": self.pooling_ks,
            "spd": self.spd,
            "pad_spd": self.pad_spd
        }
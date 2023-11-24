import math

import torch.nn as nn

from models.base_model import BaseModel
from modules import get_convolution
from torch.nn.modules.utils import _pair

class SimpleCNN(BaseModel):
    def __init__(
        self,
        img_size,
        num_classes,
        num_layers=3,
        in_channels=3,
        num_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        separable=False,
        num_skip_layers=None,
        gabor=False,
        activation="relu"
    ):
        super(SimpleCNN, self).__init__(img_size, activation)
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.separable = separable
        self.num_skip_layers=num_skip_layers
        self.gabor=gabor

        assert not(gabor and separable)

        # Define the first convolutional layer
        self.conv1 = get_convolution(
            in_channels, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers, gabor
        )
        self.softmax = nn.Softmax(dim=-1)

        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.extra_conv_layers.append(
                get_convolution(
                    num_channels, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers, gabor
                )
            )

        # Fully connected layer
        res_size = [*self.img_size]
        for l in range(self.num_layers):
            res_size[0] = math.floor(
                (res_size[0]+2*self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
            )
            res_size[1] = math.floor(
                (res_size[1]+2*self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
            )
        self.fc = nn.Linear(num_channels * res_size[0] * res_size[1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self._activation_func(x)

        for conv_layer in self.extra_conv_layers:
            x = conv_layer(x)
            x = self._activation_func(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x

    def get_sequential(self): # TODO: how to add skip connections here?
        seq = nn.Sequential()
        seq.append(self.conv1)
        seq.append(self._activation_func)

        for conv_layer in self.extra_conv_layers:
            seq.append(conv_layer)
            seq.append(self._activation_func)
        seq.append(nn.Flatten())
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
            "in_channels": self.in_channels,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "separable": self.separable,
            "num_skip_layers": self.num_skip_layers,
            "gabor": self.gabor,
        }
import math
from os import path

import torch
import torch.nn as nn
import yaml

from models.base_model import BaseModel

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
        dilation=1,
        separable=False,
        skip_connections=0
    ):
        super(SimpleCNN, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.separable = separable
        self.skip_connections=skip_connections

        # Define the first convolutional layer
        self.conv1 = self.get_convolution(
            in_channels, num_channels, kernel_size, stride, dilation, separable
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.extra_conv_layers.append(
                self.get_convolution(
                    num_channels, num_channels, kernel_size, stride, dilation, separable
                )
            )

        # Fully connected layer
        res_size = self.img_size
        for l in range(self.num_layers):
            res_size = math.floor(
                (res_size - dilation * (kernel_size - 1) - 1) / stride + 1
            )
        self.fc = nn.Linear(num_channels * res_size**2, num_classes)

    def get_convolution(
        self, in_channels, num_channels, kernel_size, stride, dilation, separable=False
    ):
        if separable:
            return SeparableConv2d(
                in_channels, num_channels, kernel_size, stride, dilation=dilation
            )
        else:
            return nn.Conv2d(
                in_channels, num_channels, kernel_size, stride, dilation=dilation
            )

    def forward(self, x):
        skip_count = 1
        if(self.skip_connections > 0):
            pre_skip = x
        x = self.conv1(x)
        x = self.relu(x)

        for conv_layer in self.extra_conv_layers:
            x = conv_layer(x)
            x = self.relu(x)
            if skip_count == self.skip_connections:
                x = x+self.center_crop(pre_skip, x.shape)
                skip_count=0
            else:
                skip_count +=1

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x
    
    def center_crop(self, x, shape):
        shape_diff = shape - x.shape
        starts = shape_diff // 2
        return x[starts[0]:shape[0],starts[1]:shape[1],starts[2]:shape[2],starts[3]:shape[3]]

    def get_sequential(self): # TODO: add skip connections here?
        seq = nn.Sequential()
        seq.append(self.conv1)
        seq.append(self.relu)

        for conv_layer in self.extra_conv_layers:
            seq.append(conv_layer)
            seq.append(self.relu)
        seq.append(nn.Flatten())
        seq.append(self.fc)
        seq.append(self.softmax)
        return seq

    def config(self) -> dict:
        return {
            "type" : "simple",
            "config" : {
            "img_size": self.img_size,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "in_channels": self.in_channels,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "separable": self.separable,
            "skip_connections": self.skip_connections,
        }}


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(SeparableConv2d, self).__init__()
        self.vertical_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            bias=bias,
        )
        self.horizontal_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            bias=bias,
        )

    def forward(self, x):
        x = self.vertical_conv(x)
        x = self.horizontal_conv(x)
        return x

import math
from os import path

import torch
import torch.nn as nn
import yaml
from abc import ABC

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
        padding=0,
        dilation=1,
        separable=False,
        num_skip_layers=None
    ):
        super(SimpleCNN, self).__init__(img_size)
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.separable = separable
        self.num_skip_layers=num_skip_layers

        # Define the first convolutional layer
        self.conv1 = get_convolution(
            in_channels, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.extra_conv_layers.append(
                get_convolution(
                    num_channels, num_channels, kernel_size, stride, padding, dilation, separable, num_skip_layers
                )
            )

        # Fully connected layer
        res_size = self.img_size
        for l in range(self.num_layers):
            res_size = math.floor(
                (res_size+2*padding - dilation * (kernel_size - 1) - 1) / stride + 1
            )
        self.fc = nn.Linear(num_channels * res_size**2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        for conv_layer in self.extra_conv_layers:
            x = conv_layer(x)
            x = self.relu(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x

    def get_sequential(self): # TODO: how to add skip connections here?
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
            "padding": self.padding,
            "dilation": self.dilation,
            "separable": self.separable,
            "num_skip_layers": self.num_skip_layers,
        }}

def get_convolution(in_channels, num_channels, kernel_size, stride, padding, dilation, separable=False, num_skip_layers=None):
    if num_skip_layers is not None:
        return ResConv2d(in_channels, num_channels, kernel_size, num_skip_layers, stride, padding, dilation= dilation, separable=separable)
    elif separable:
        return SeparableConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    else:
        return nn.Conv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    
class ModConv2d(nn.Module, ABC):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
    ):
        super(ModConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride=(stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding=padding
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        self.bias = bias

class SeparableConv2d(ModConv2d):
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
        super(SeparableConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

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


class ResConv2d(ModConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layers = None,
        stride=1,
        padding=0,
        dilation=1,
        separable=False,
        bias=True,
    ):
        super(ResConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        self.layers = layers
        self.separable = separable
        
        single_kernel_size = kernel_size//layers+1 # TODO: Attention - works only if num_skip_layers and kernel_size match! Implement check or generalization
        

        self.stacked_convs = nn.Sequential()
        self.stacked_convs.append(get_convolution(in_channels, out_channels, single_kernel_size, stride, padding, dilation*layers - 1, separable=separable))
        for _ in range(layers-1):
            self.stacked_convs.append(get_convolution(out_channels, out_channels, single_kernel_size, stride=1, padding=0, dilation=1, separable=separable))

    def center_crop(self, x, shape):
        shape_diff = [x.shape[i] - shape[i] for i in range(len(shape))]
        starts = [shape_diff[i] // 2 for i in range(len(shape))]
        return x[starts[0]:starts[0]+shape[0],starts[1]:starts[1]+shape[1],starts[2]:starts[2]+shape[2],starts[3]:starts[3]+shape[3]]

    def forward(self, x):
        # TODO: different "cropping" mechanisms, inner activation functions?
        out = self.stacked_convs(x)
        return out + self.center_crop(x, out.shape)

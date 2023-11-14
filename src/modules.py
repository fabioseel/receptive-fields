import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
from abc import ABC

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
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding) # TODO: implement padding_mode
        self.dilation = _pair(dilation)
        self.bias = bias

class GaborConv2d(ModConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros" # TODO: implement padding_mode
    ):
        """
        Convolutional layer described by a Gabor, only the gabor parameters are learnable
        https://github.com/iKintosh/GaborNet/blob/master/GaborNet/GaborLayer.py
        """
        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.groups = groups
        self.is_calculated = False

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = Parameter(torch.zeros(out_channels), requires_grad=True)

        # small addition to avoid division by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=True
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=True
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ],
            indexing="ij"
        )
        self.y = Parameter(self.y.clone(), requires_grad=True)
        self.x = Parameter(self.x.clone(), requires_grad=True)

        self.weight = Parameter(
            torch.empty((out_channels, in_channels//groups, self.kernel_size[0], self.kernel_size[1])),
            requires_grad=True
        )

    def requires_grad(self):
        return any(param.requires_grad for param in self.parameters())

    def forward(self, input_tensor):
        if self.training and self.requires_grad():
            self.calculate_weights()
            self.is_calculated = False
        if not self.training or not self.requires_grad():
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return torch.nn.functional.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def calculate_weights2(self):
        for i in range(self.weight.shape[0]):
            for j in range(self.weight.shape[1]):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.weight.data[i, j] = g

    def calculate_weights(self):
        sigma = self.sigma[:,:,None,None].expand(-1,-1,*self.y.shape)
        freq = self.freq[:,:,None,None].expand(-1,-1,*self.y.shape)
        theta = self.theta[:,:,None,None].expand(-1,-1,*self.y.shape)
        psi = self.psi[:,:,None,None].expand(-1,-1,*self.y.shape)

        rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
        roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

        g = torch.exp(
            -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
        )
        g = g * torch.cos(freq * rotx + psi)
        g = g / (2 * math.pi * sigma ** 2)
        self.weight.data = g

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

def get_convolution(in_channels, num_channels, kernel_size, stride, padding, dilation, separable=False, num_skip_layers=None, gabor=False):
    if num_skip_layers is not None:
        return ResConv2d(in_channels, num_channels, kernel_size, num_skip_layers, stride, padding, dilation= dilation, separable=separable)
    elif separable:
        return SeparableConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    elif gabor:
        return GaborConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    else:
        return nn.Conv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )

class L2Pool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)

        self.kernel_size = self.pool.kernel_size
        self.stride = self.pool.stride
        self.padding = self.pool.padding
        self.ceil_mode = self.pool.ceil_mode
        self.count_include_pad = self.pool.count_include_pad
        self.divisor_override = self.pool.divisor_override
        
    def forward(self, x):
        return torch.sqrt(self.pool(x ** 2))
import math
import torch
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
from torch.nn.modules.utils import _pair

class GaborConv2d(Module):
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
        padding_mode="zeros",
    ):
        """
        Convolutional layer described by a Gabor, only the gabor parameters are learnable
        https://github.com/iKintosh/GaborNet/blob/master/GaborNet/GaborLayer.py
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding) # TODO: implement padding_mode
        self.dilation = _pair(dilation)
        self.groups = groups
        self.is_calculated = False

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = Parameter(torch.zeros(out_channels), requires_grad=False)

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
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ],
            indexing="ij"
        )
        self.y = Parameter(self.y, requires_grad=False)
        self.x = Parameter(self.x, requires_grad=False)

        self.weight = Parameter(
            torch.empty((out_channels, in_channels//groups, self.kernel_size[0], self.kernel_size[1])),
            requires_grad=False
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
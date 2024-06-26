import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
from lkan.models import KANConv2d
from abc import ABC
import copy


class NamedFromSequential(nn.Module):
    """
    When initialized with a torch.nn.Sequential Module, adds each of the modules in sequential as individual attributes to this module.
    Helpful for usage with libraries that allow to access specific model layers via the attribute names
    """

    def __init__(self, seq_model: nn.Sequential):
        super().__init__()
        self.module_names = []
        mod_count = {}
        for module in seq_model:
            mod_name = str(module.__class__)[:-2].split(".")[-1]
            if mod_name not in mod_count.keys():
                mod_count[mod_name] = 0
            else:
                mod_count[mod_name] += 1
            attr_name = mod_name + "_" + str(mod_count[mod_name])
            self.__setattr__(attr_name, copy.deepcopy(module))
            self.module_names.append(attr_name)

    def forward(self, x):
        for attr_name in self.module_names:
            x = self.__getattr__(attr_name)(x)
        return x


class ModConv2d(nn.Module, ABC):
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
        super(ModConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)  # TODO: implement padding_mode
        self.dilation = _pair(dilation)
        self.bias = bias


class SymConv2d(ModConv2d):
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
        super(SymConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )
        assert out_channels % 4 == 0  # assert that each conv can be mirrored / rotated
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels // 4,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups,
            self.bias,
            padding_mode
        )

    def forward(self, input_tensor: torch.tensor):
        assert len(input_tensor.shape) == 4
        return torch.concatenate(
            [
                self.conv(input_tensor),
                self.conv(input_tensor.flip(2)).flip(2),
                self.conv(input_tensor.transpose(2, 3)).transpose(3, 2),
                self.conv(input_tensor.transpose(2, 3).flip(3)).flip(3).transpose(3, 2),
            ],
            dim=1,
        )


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
        padding_mode="zeros",  # TODO: implement padding_mode
    ):
        """
        !!! not working, some problem with the gradient apparently!!!
        Convolutional layer described by a Gabor, only the gabor parameters are learnable
        https://github.com/iKintosh/GaborNet/blob/master/GaborNet/GaborLayer.py
        """
        super(GaborConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )
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
            indexing="ij",
        )
        self.y = Parameter(self.y.clone(), requires_grad=True)
        self.x = Parameter(self.x.clone(), requires_grad=True)

        self.weight = Parameter(
            torch.empty(
                (
                    out_channels,
                    in_channels // groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                )
            ),
            requires_grad=True,
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
        return torch.nn.functional.conv2d(
            input_tensor,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def calculate_weights2(self):
        for i in range(self.weight.shape[0]):
            for j in range(self.weight.shape[1]):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(-0.5 * ((rotx**2 + roty**2) / (sigma + self.delta) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma**2)
                self.weight.data[i, j] = g

    def calculate_weights(self):
        sigma = self.sigma[:, :, None, None].expand(-1, -1, *self.y.shape)
        freq = self.freq[:, :, None, None].expand(-1, -1, *self.y.shape)
        theta = self.theta[:, :, None, None].expand(-1, -1, *self.y.shape)
        psi = self.psi[:, :, None, None].expand(-1, -1, *self.y.shape)

        rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
        roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

        g = torch.exp(-0.5 * ((rotx**2 + roty**2) / (sigma + self.delta) ** 2))
        g = g * torch.cos(freq * rotx + psi)
        g = g / (2 * math.pi * sigma**2)
        self.weight.data = g


class StaggeredStridedConv2d(ModConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        """
        A convolution that if a stride is applied will divide the convolution into several ones:
        The product of the stride $n$ determines how many convolutions will be applied.
        Each of the resulting convolutions produces $n_out_channels/n$ outputs.
        For each of these convolutions the starting point is changed, such that they become misaligned and even though stride is applied,
        effectively only a single pixel is skipped between the convolutions.
        """
        super(StaggeredStridedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )
        self.n_shifts_x = self.stride[0]
        self.n_shifts_y = self.stride[1]
        n_shifts_total = self.n_shifts_x * self.n_shifts_y
        n_channels_per_conv = out_channels // n_shifts_total
        remain_channels = out_channels - (n_shifts_total - 1) * n_channels_per_conv
        self.pad = nn.ZeroPad2d(
            (
                self.n_shifts_x // 2 + self.padding[0],
                self.n_shifts_x // 2 + self.padding[0],
                self.n_shifts_y // 2 + self.padding[1],
                self.n_shifts_y // 2 + self.padding[1],
            )
        )
        # TODO: Unite "Paddings"
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    remain_channels,
                    self.kernel_size,
                    self.stride,
                    padding=0,
                    dilation=self.dilation,
                    bias=self.bias,
                    padding_mode=padding_mode,
                )
            ]
        )
        for i in range(n_shifts_total - 1):
            self.convs.append(
                nn.Conv2d(
                    self.in_channels,
                    n_channels_per_conv,
                    self.kernel_size,
                    self.stride,
                    padding=0,
                    dilation=self.dilation,
                    bias=self.bias,
                    padding_mode=padding_mode,
                )
            )

    def forward(self, x):
        out = []
        eff_in_size = (
            x.shape[-2] + self.padding[0] * 2,
            x.shape[-1] + self.padding[1] * 2,
        )
        x = self.pad(x)
        # eff_in_size = (x.shape[-2]-self.stride[0],x.shape[-1]-self.stride[1])
        for x_shift in range(self.stride[0]):
            for y_shift in range(self.stride[1]):
                cur_inp = x[
                    :,
                    :,
                    x_shift : x_shift + eff_in_size[0],
                    y_shift : y_shift + eff_in_size[1],
                ]
                out.append(self.convs[x_shift + (self.stride[0]) * y_shift](cur_inp))
        x = torch.concat(out, dim=1)
        return x


class DepthwiseSeparableConv2d(ModConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(DepthwiseSeparableConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )
        self.depth_conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False,
            groups=in_channels,
            padding_mode=padding_mode,
        )
        self.point_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor, "stolen" from https://github.com/Cateners/yolov8-spd/commit/6e632246ba71e9932512eee834117c343e45b614#diff-70b1aad8c0068184a0e145431a251d90d480aed0226918c90c5a8ccd5a84f0f8
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )


class SpaceToDepth(nn.Module):
    def __init__(self, dimension=1, factor=2, old_spd_reorder=False, pad=True):
        """
        Rearranges an input tensor such that the output size is scaled by factor and instead the pixels are rearranged into the channel domain.
        Output will have size (bs, in_channels*factor^2, w/factor, h/factor). Pads zeros if needed
        """
        super().__init__()
        self.d = dimension
        self.factor = factor
        self.old_spd_reorder = old_spd_reorder
        self.pad = pad

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        new_height = height // self.factor
        new_width = width // self.factor

        if self.pad:
            # Calculate padding
            if height % self.factor != 0:
                new_height += 1
                pad_height = max(0, new_height * self.factor - height)
            else:
                pad_height = 0

            if width % self.factor != 0:
                new_width += 1
                pad_width = max(0, new_width * self.factor - width)
            else:
                pad_width = 0

            # Pad the input tensor
            x = nn.functional.pad(x, (0, pad_width, 0, pad_height))
        else:
            # Cut the input tensor
            x = x[:, :, : new_height * self.factor, : new_width * self.factor]

        # Reshape the tensor
        x = x.view(
            batch_size, channels, new_height, self.factor, new_width, self.factor
        )
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(
            batch_size, channels * self.factor * self.factor, new_height, new_width
        )

        if self.old_spd_reorder:
            assert self.factor == 2
            assert self.d == 1
            old_order = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]
            x = x[..., old_order, :, :]

        return x


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
        super(SeparableConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

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
        layers=None,
        stride=1,
        padding=0,
        dilation=1,
        separable=False,
        bias=True,
    ):
        super(ResConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

        self.layers = layers
        self.separable = separable

        single_kernel_size = (
            kernel_size // layers + 1
        )  # TODO: Attention - works only if num_skip_layers and kernel_size match! Implement check or generalization

        self.stacked_convs = nn.Sequential()
        self.stacked_convs.append(
            get_convolution(
                in_channels,
                out_channels,
                single_kernel_size,
                stride,
                padding,
                dilation * layers - 1,
                separable=separable,
            )
        )
        for _ in range(layers - 1):
            self.stacked_convs.append(
                get_convolution(
                    out_channels,
                    out_channels,
                    single_kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    separable=separable,
                )
            )

    def center_crop(self, x, shape):
        shape_diff = [x.shape[i] - shape[i] for i in range(len(shape))]
        starts = [shape_diff[i] // 2 for i in range(len(shape))]
        return x[
            starts[0] : starts[0] + shape[0],
            starts[1] : starts[1] + shape[1],
            starts[2] : starts[2] + shape[2],
            starts[3] : starts[3] + shape[3],
        ]

    def forward(self, x):
        # TODO: different "cropping" mechanisms, inner activation functions?
        out = self.stacked_convs(x)
        return out + self.center_crop(x, out.shape)


def get_convolution(
    in_channels,
    num_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    separable=False,
    num_skip_layers=None,
    gabor=False,
    misaligned=False,
    symconv = False,
    kanconv = False
):
    if num_skip_layers is not None:
        return ResConv2d(
            in_channels,
            num_channels,
            kernel_size,
            num_skip_layers,
            stride,
            padding,
            dilation=dilation,
            separable=separable,
        )
    elif separable:
        return SeparableConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    elif gabor:
        return GaborConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    elif misaligned:
        return StaggeredStridedConv2d(
            in_channels, num_channels, kernel_size, stride, padding, dilation=dilation
        )
    elif symconv:
        return SymConv2d(in_channels, num_channels, kernel_size, stride, padding, dilation=dilation)
    elif kanconv:
        return KANConv2d(in_channels, num_channels, kernel_size, stride, padding[0], dilation=dilation[0])
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
        return torch.sqrt(self.pool(x**2))

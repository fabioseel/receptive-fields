import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.simple import SeparableConv2d, ResConv2d, ModConv2d


def normalizeZeroOne(input):
    return (input - input.min()) / (input.max() - input.min())

def sum_collapse_output(out_tensor):
    if len(out_tensor.shape) > 2:
        sum_dims = [2+i for i in range(len(out_tensor.shape)-2)]
        out_tensor = torch.sum(out_tensor, dim=sum_dims)
    return out_tensor


def _dataset_average(
    model: nn.Module, dataloader: DataLoader, desired_output=torch.Tensor, device=None
):
    model.eval()  # Set the model to evaluation mode
    weighted_sum = torch.zeros(next(iter(dataloader))[0].shape[1:], device=device)

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs=inputs.to(device)
            outputs = model(inputs)

            # Calculate the error as the difference between the predicted and desired distributions
            error = torch.sum(torch.pow((desired_output - outputs), 4))
            weight = 1.0 / (1.0 + error)
            weighted_inputs = inputs * weight
            weighted_sum += weighted_inputs.sum(dim=0)

    return normalizeZeroOne(weighted_sum.cpu())


def dataset_average(
    model: nn.Module, dataloader: DataLoader, device=None
):
    num_outputs, input_size = get_input_output_shape(model)
    results = torch.zeros((num_outputs, *dataloader.dataset[0][0].shape))
    for i in tqdm(range(num_outputs)):
        output_signal = torch.zeros(num_outputs, device=device)
        output_signal[i] = 1
        results[i] = _dataset_average(
            model, dataloader, output_signal, device
        )
    return results

def remove_padding(model: nn.Sequential):
    new_model = nn.Sequential()
    linear = [isinstance(module, nn.Linear) for module in model]
    if not np.any(linear):
        for module in model:
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                     padding=0, dilation=module.dilation, groups=module.groups)
                new_conv.weight = module.weight
                new_conv.bias = module.bias
                new_model.append(new_conv)
            elif isinstance(module, SeparableConv2d):
                new_conv = SeparableConv2d(module.vertical_conv.in_channels, module.horizontal_conv.out_channels, module.vertical_conv.kernel_size[0], module.vertical_conv.stride[0],
                                     padding=0, dilation=module.vertical_conv.dilation[0])
                new_conv.vertical_conv.weight = module.vertical_conv.weight
                new_conv.vertical_conv.bias = module.vertical_conv.bias
                new_conv.horizontal_conv.weight = module.horizontal_conv.weight
                new_conv.horizontal_conv.bias = module.horizontal_conv.bias
                new_model.append(new_conv)
            elif isinstance(module, ResConv2d):
                new_conv = ResConv2d(module.in_channels,module.out_channels,module.kernel_size[0],module.layers, module.stride, padding=0, dilation=module.dilation[0], separable=module.separable)
                for (_old_conv, _new_conv) in zip(module.stacked_convs, new_conv.stacked_convs):
                    _new_conv.weight = _old_conv.weight
                    _new_conv.bias = _old_conv.bias
                new_model.append(new_conv)
            else:
                new_model.append(module)
    else:
        new_model = model
    
    return new_model

def effective_receptive_field(model: nn.Sequential, n_batch: int = 2048, fill_value: float=None, rf_size: tuple=None, device=None):
    '''
    if fill value is given a single 'empty' input of that value is used
    '''
    model = remove_padding(model).to(device)
    num_outputs, input_size = get_input_output_shape(model)
    if rf_size is not None:
        input_size=rf_size
    results = torch.zeros((num_outputs, *input_size))
    for i in tqdm(range(num_outputs)):
        output_signal = torch.zeros(num_outputs, device=device)
        output_signal[i] = 1
        results[i] = single_effective_receptive_field(
            model, output_signal, input_size, n_batch, fill_value, device
        )
    return results


def single_effective_receptive_field(
    model: nn.Sequential,
    output_signal: torch.tensor,
    input_size: torch.Size,
    n_batch: int = 2048,
    fill_value: float =None,
    device = None
):
    '''
    check the doc of the parent method (effective_receptive_field) for reference
    '''
    model.eval()
    if fill_value is not None:
        input_tensor = torch.full((1, *input_size), fill_value=fill_value, requires_grad=True, device=device)
    else:
        input_tensor = torch.randn((n_batch, *input_size), requires_grad=True, device=device)
    output = model(input_tensor)
    output = sum_collapse_output(output)
    target_output = output * output_signal
    target_output.sum().backward()
    eff_rf = input_tensor.grad.sum(0)
    return eff_rf.cpu().detach()


def _activation_triggered_average(model: nn.Module, n_batch: int = 2048, rf_size=None, device=None):
    model.eval()
    if rf_size is None:
        _out_channels, input_size = get_input_output_shape(model)
    else:
        input_size = rf_size
    input_tensor = torch.randn((n_batch, *input_size), requires_grad=False, device=device)
    output = model(input_tensor)
    output = sum_collapse_output(output)
    input_tensor = input_tensor[:, None, :, :, :].expand(
        -1, output.shape[1], -1, -1, -1
    )

    weights = output[:, :, None, None, None].expand(-1, -1, *input_size)
    weight_sums = output.abs().sum(0)
    weight_sums[weight_sums == 0] = 1
    weighted = (weights.abs() * input_tensor).sum(0) / weight_sums[:, None, None, None]
    return weighted.cpu().detach() / n_batch


def activation_triggered_average(
    model: nn.Module, n_batch: int = 2048, n_iter: int = 1, rf_size=None, device=None
): # TODO: Buggy? Returns only noise
    weighted = _activation_triggered_average(model, n_batch, device=device)
    for _ in tqdm(range(n_iter - 1), total=n_iter, initial=1):
        weighted += _activation_triggered_average(model, n_batch, rf_size, device=device)
    return weighted.cpu().detach() / n_iter


def get_input_output_shape(model: nn.Sequential):
    _first = 0
    down_stream_linear = False
    for layer in reversed(model):
        _first += 1
        if isinstance(layer, nn.Linear):
            num_outputs = layer.out_features
            in_channels = 1
            in_size = layer.in_features
            down_stream_linear = True
            break
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, ModConv2d): # TODO: Refactor, nicify
            num_outputs = layer.out_channels
            in_channels = layer.in_channels
            in_size = layer.in_channels * ((layer.kernel_size[0]-1)*layer.dilation[0]+1) ** 2
            break
        # elif isinstance(layer, SeparableConv2d):
        #     num_outputs = layer.horizontal_conv.out_channels
        #     in_channels = layer.vertical_conv.in_channels
        #     in_size = layer.vertical_conv.in_channels * layer.vertical_conv.kernel_size[0] ** 2
        #     break

    for layer in reversed(model[:-_first]):
        if isinstance(layer, nn.Linear):
            in_channels = 1
            in_size = layer.in_features
            down_stream_linear = True
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, ModConv2d):
            in_channels = layer.in_channels
            in_size = math.sqrt(in_size / layer.out_channels)
            in_size = (
                (in_size - 1) * layer.stride[0] * layer.dilation[0]
                - 2 * layer.padding[0] * down_stream_linear
                + layer.kernel_size[0]
            )
            in_size = in_size**2 * in_channels
        # elif isinstance(layer, SeparableConv2d):
        #     in_channels = layer.vertical_conv.in_channels
        #     in_size = math.sqrt(in_size / layer.horizontal_conv.out_channels)
        #     in_size = (
        #         (in_size - 1) * layer.vertical_conv.stride[0] * layer.vertical_conv.dilation[0]
        #         - 2 * layer.vertical_conv.padding[0] * down_stream_linear
        #         + layer.vertical_conv.kernel_size[0]
        #     )
        #     in_size = in_size**2 * in_channels

    in_size = math.floor(math.sqrt(in_size / in_channels))
    input_size = (in_channels, in_size, in_size)
    return num_outputs, input_size


def backprop_maximization(
    model: nn.Module,
    n_iter: int = 2048,
    batch_size=16,
    reduction=True,
    smoothened=False,
    device=None
):
    num_outputs, input_size = get_input_output_shape(model)
    results = torch.zeros((num_outputs, *input_size))
    for i in tqdm(range(num_outputs)):
        output_signal = torch.zeros(num_outputs)
        output_signal[i] = 1
        results[i] = single_backprop_maximization(
            model, output_signal, input_size, n_iter, batch_size, reduction, smoothened, device
        )
    return results


def single_backprop_maximization(
    model: nn.Module,
    output_signal: torch.tensor,
    input_size: torch.Size,
    n_iter: int = 2048,
    batch_size=16,
    reduction=True,
    smoothened=False,
    device=None
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    smooth_loss = nn.MSELoss()

    input_tensor = torch.randn((batch_size, *input_size), requires_grad=True, device=device)
    for i in tqdm(range(n_iter)):
        input_tensor.requires_grad = True
        output = model(input_tensor.repeat(1, 1, 1, 1))
        loss = criterion(output, output_signal.to(device).repeat(batch_size, 1))
        if smoothened and i > n_iter / 3:
            loss = (
                loss
                + smooth_loss(input_tensor[..., 1:, 1:], input_tensor[..., :-1, :-1])
                + smooth_loss(input_tensor[..., 1:, :-1], input_tensor[..., :-1, 1:])
                + smooth_loss(input_tensor[..., 1:, :], input_tensor[..., :-1, :])
                + smooth_loss(input_tensor[..., :, 1:], input_tensor[..., :, :-1])
            )
        loss.backward()
        input_tensor = normalizeZeroOne(input_tensor.detach() - input_tensor.grad)
    if reduction:
        return normalizeZeroOne(input_tensor.mean(0)).cpu().detach()
    else:
        return input_tensor.cpu().detach()

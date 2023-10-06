import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def normalizeZeroOne(input):
    return (input - input.min()) / (input.max() - input.min())


def dataset_average(
    model: nn.Module, dataloader: DataLoader, desired_output=torch.Tensor
):
    model.eval()  # Set the model to evaluation mode
    weighted_sum = torch.zeros(next(iter(dataloader))[0].shape[1:])

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)

            # Calculate the error as the difference between the predicted and desired distributions
            error = torch.sum(torch.pow((desired_output - outputs), 4))
            weight = 1.0 / (1.0 + error)
            weighted_inputs = inputs * weight
            weighted_sum += weighted_inputs.sum(dim=0)

    return normalizeZeroOne(weighted_sum)


def effective_receptive_field(model: nn.Module, n_batch: int = 2048):
    num_outputs, input_size = get_input_output_shape(model)
    results = torch.zeros((num_outputs, *input_size))
    for i in tqdm(range(num_outputs)):
        output_signal = torch.zeros(num_outputs)
        output_signal[i] = 1
        results[i] = single_effective_receptive_field(
            model, output_signal, input_size, n_batch
        )
    return results


def single_effective_receptive_field(
    model: nn.Module,
    output_signal: torch.tensor,
    input_size: torch.Size,
    n_batch: int = 2048,
):
    model.eval()
    input_tensor = torch.randn((n_batch, *input_size), requires_grad=True)
    output = model(input_tensor)
    target_output = output * output_signal
    target_output.sum().backward()
    eff_rf = input_tensor.grad.sum(0)
    return normalizeZeroOne(eff_rf)


def _activation_triggered_average(model: nn.Module, n_batch: int = 2048):
    model.eval()
    _out_channels, input_size = get_input_output_shape(model)
    input_tensor = torch.randn((n_batch, *input_size), requires_grad=False)
    output = model(input_tensor)
    output = output.squeeze()
    input_tensor = input_tensor[:, None, :, :, :].expand(
        -1, output.shape[1], -1, -1, -1
    )

    weights = output[:, :, None, None, None].expand(-1, -1, *input_size)
    weight_sums = output.abs().sum(0)
    weight_sums[weight_sums == 0] = 1
    weighted = (weights.abs() * input_tensor).sum(0) / weight_sums[:, None, None, None]
    return weighted.detach() / n_batch


def activation_triggered_average(
    model: nn.Module, n_batch: int = 2048, n_iter: int = 1
):
    weighted = _activation_triggered_average(model, n_batch)
    for _ in tqdm(range(n_iter - 1), total=n_iter, initial=1):
        weighted += _activation_triggered_average(model, n_batch)
    return weighted.detach() / n_iter


def get_input_output_shape(model: nn.Sequential):
    _first = 0
    for layer in reversed(model):
        _first += 1
        if isinstance(layer, nn.Linear):
            num_outputs = layer.out_features
            in_channels = 1
            in_size = layer.in_features
            break
        elif isinstance(layer, nn.Conv2d):
            num_outputs = layer.out_channels
            in_channels = layer.in_channels
            in_size = layer.in_channels * layer.kernel_size[0] ** 2
            break

    for layer in reversed(model[:-_first]):
        if isinstance(layer, nn.Linear):
            in_channels = 1
            in_size = layer.in_features
        elif isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
            in_size = math.sqrt(in_size / layer.out_channels)
            in_size = (
                (in_size - 1) * layer.stride[0]
                - 2 * layer.padding[0]
                + layer.kernel_size[0]
            )
            in_size = in_size**2 * in_channels

    in_size = math.floor(math.sqrt(in_size / in_channels))
    input_size = (in_channels, in_size, in_size)
    return num_outputs, input_size


def backprop_maximization(
    model: nn.Module,
    n_iter: int = 2048,
    batch_size=16,
    reduction=True,
    smoothened=False,
):
    num_outputs, input_size = get_input_output_shape(model)
    results = torch.zeros((num_outputs, *input_size))
    for i in tqdm(range(num_outputs)):
        output_signal = torch.zeros(num_outputs)
        output_signal[i] = 1
        results[i] = single_backprop_maximization(
            model, output_signal, input_size, n_iter, batch_size, reduction, smoothened
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
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    smooth_loss = nn.MSELoss()

    input_tensor = torch.randn((batch_size, *input_size), requires_grad=True)
    for i in tqdm(range(n_iter)):
        input_tensor.requires_grad = True
        output = model(input_tensor.repeat(1, 1, 1, 1))
        loss = criterion(output, output_signal.repeat(batch_size, 1))
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
        return normalizeZeroOne(input_tensor.mean(0)).detach()
    else:
        return input_tensor.detach()

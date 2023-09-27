from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

def normalizeZeroOne(input):
    return (input - input.min()) / (input.max()-input.min())

def dataset_average(model: nn.Module, dataloader: DataLoader, desired_output = torch.Tensor):
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

def effective_receptive_field(model:nn.Module, output_signal:torch.tensor, input_size:torch.Size, n_iter:int=2048):
    model.eval()
    input_tensor = torch.randn((n_iter, *input_size), requires_grad=True)
    output = model(input_tensor)
    target_output = output * output_signal
    target_output.sum().backward()
    eff_rf = input_tensor.grad.sum(0)
    return normalizeZeroOne(eff_rf)

def backprop_maximization(model:nn.Module, output_signal:torch.tensor, input_size:torch.Size, n_iter:int=2048, batch_size=16, reduction=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    input_tensor = torch.randn((batch_size,*input_size), requires_grad=True)
    for i in range(n_iter):
        input_tensor.requires_grad = True
        output = model(input_tensor.repeat(1,1,1,1))
        loss = criterion(output, output_signal.repeat(batch_size,1))
        loss.backward()
        input_tensor = normalizeZeroOne(input_tensor.detach()-input_tensor.grad)
    if reduction:
        return normalizeZeroOne(input_tensor.mean(0)).detach()
    else:
        return input_tensor.detach()
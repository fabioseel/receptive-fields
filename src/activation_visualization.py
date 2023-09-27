from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

def dataset_average(model: nn.Module, dataloader: DataLoader, desired_output = torch.Tensor):
    model.eval()  # Set the model to evaluation mode
    total_weight = 0.0
    weighted_sum = torch.zeros(next(iter(dataloader))[0].shape[1:])

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            
            # Calculate the error as the difference between the predicted and desired distributions
            error = torch.sum(torch.pow((desired_output - outputs), 4))
            weight = 1.0 / (1.0 + error)
            weighted_inputs = inputs * weight
            weighted_sum += weighted_inputs.sum(dim=0)
            total_weight += weight.sum(dim=0)

    return weighted_sum / total_weight

def effective_receptive_field(model:nn.Module, output_signal:torch.tensor, input_size:torch.Size, n_iter:int=2048):
    model.eval()
    input_tensor = torch.randn((n_iter, *input_size), requires_grad=True)
    output = model(input_tensor)
    target_output = output * output_signal
    target_output.sum().backward()
    eff_rf = input_tensor.grad.sum(0)
    return (eff_rf - eff_rf.min()) / (eff_rf.max()-eff_rf.min())
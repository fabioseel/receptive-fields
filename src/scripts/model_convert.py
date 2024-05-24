from retinal_rl_models.encoder import GenericModel
from receptive_fields.models.simple import SimpleCNN
import argparse
import torch
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)

args = parser.parse_args()

in_model = SimpleCNN.load(args.input)

# Check compatibility
assert in_model.separable == False
assert in_model.num_skip_layers == None
assert in_model.gabor == False
assert in_model.staggered == False
assert in_model.spd == 1
assert in_model.symconv == False
assert in_model.kanconv == False

if in_model.num_fc_layers != 1:
    warnings.warn("in_model has more than 1 FC layer, will be ignored atm", )

tuple_warning = "{name} is a tuple: {shape} - the new model will use only the larger of the values. Might result in error since output shape of conv might be changed."

padding = in_model.padding
if isinstance(padding, tuple):
    warnings.warn(tuple_warning.format(name="padding", shape=padding))
    padding = max(in_model.padding)

dilation = in_model.dilation
if isinstance(dilation, tuple):
    warnings.warn(tuple_warning.format(name="padding", shape=dilation))
    dilation = max(in_model.dilation)

out_model = GenericModel(
    inp_shape=(in_model.in_channels, *in_model.img_size),
    out_size=in_model.fc_dim,
    num_layers=in_model.num_layers,
    fc_in_size=in_model.fc_in_size,
    num_channels=in_model.num_channels,
    kernel_size=in_model.kernel_size,
    stride=in_model.stride,
    padding=padding,
    dilation=dilation,
    act_name=in_model.activation,
    pooling_ks=in_model.pooling_ks
)

with torch.no_grad():
    for i, layer in enumerate(in_model.get_sequential()):
        if isinstance(layer, torch.nn.Conv2d):
            out_model.conv_head[i].weight.copy_(layer.weight)
            out_model.conv_head[i].bias.copy_(layer.bias)
        if isinstance(layer, torch.nn.Linear):
            out_model.fc.weight.copy_(layer.weight)
            out_model.fc.bias.copy_(layer.bias)
            break

out_model.save(args.output)

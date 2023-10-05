from retinal_rl.analysis import statistics as ret_stats
from activation_visualization import get_input_output_shape
import torch

class Object(object):
    pass

def pack_input(model):
    cfg = Object()
    cfg.device="cpu"
    env = Object()
    _out_channels, input_size = get_input_output_shape(model)
    env.observation_space={"obs":torch.empty(input_size)}
    actor_critic=Object()
    actor_critic.encoder = torch.nn.Module()
    encoder = torch.nn.Module()
    encoder.conv_head = model
    encoder.forward = lambda x: encoder.conv_head(x)
    actor_critic.encoder.vision_model = encoder
    actor_critic.encoder.forward = lambda x: encoder.forward(x) 
    return cfg,env,actor_critic

def gaussian_noise_stas(model, n_batch, n_reps):
    return ret_stats.gaussian_noise_stas(*pack_input(model), n_batch, n_reps, prgrs=True)

def gradient_receptive_fields(model):
    """not working currently..."""
    return ret_stats.gradient_receptive_fields(*pack_input(model),prgrs=True)
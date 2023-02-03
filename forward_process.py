import torch
from utilities import *
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from utilities import *
import numpy
from noise import *





def forward_diffusion_sample(x_0, t, constant_dict, config):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod= constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod']

    noise = get_noise(x_0, config)
    device = config.model.device

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape, config)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape, config
    )
    # mean + variance
    x = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    x = x.to(device)
    noise = noise.to(device)
    return x, noise



def forward_ti_steps(t, ti, x_t_ti, x_0, beta, config):
    x_t_ti = x_t_ti.to(config.model.device)
    noise = get_noise(x_t_ti, config)
    x_t = x_t_ti + (compute_alpha(beta, t, config).sqrt() - compute_alpha(beta, t-ti, config).sqrt()) * x_0 + ((1 - compute_alpha(beta, t, config)).sqrt() - (1 - compute_alpha(beta, t-ti, config)).sqrt()) * noise
    return x_t

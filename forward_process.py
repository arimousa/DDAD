import torch
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import numpy





def forward_diffusion_sample(x_0, t, constant_dict, config):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod= constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod']

    noise = torch.randn_like(x_0).to(config.model.device)
    device = config.model.device

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape, config
    )
    # mean + variance
    x = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    x = x.to(device)
    noise = noise.to(device)
    return x, noise


def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


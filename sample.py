import torch
from noise import *
from utilities import *

@torch.no_grad()
def sample_timestep(config, model, constant_dict, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(constant_dict['betas'], t, x.shape, config)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        constant_dict['sqrt_one_minus_alphas_cumprod'], t, x.shape, config
    )
    sqrt_recip_alphas_t = get_index_from_list(constant_dict['sqrt_recip_alphas'], t, x.shape, config)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(constant_dict['posterior_variance'], t, x.shape, config)
    if t == 0:
        return model_mean
    else:
        noise = get_noise(x, t, config)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 



import torch
import os
import torch.nn as nn
from forward_process import *
from backbone import *
from noise import *


def get_loss(model, constant_dict, x_0, t, config):
    
    x_noisy, noise = forward_diffusion_sample(x_0, t , constant_dict, config)
  
    noise_pred = model(x_noisy, t)
    # loss = F.l1_loss(noise, noise_pred)
    loss = F.mse_loss(noise, noise_pred)
   # return loss



    cos_loss = torch.nn.CosineSimilarity()
    cosloss = 0
    x_0 = x_0.to(config.model.device)

  #  x_noisy, noise = forward_diffusion_sample(x_0, t , constant_dict, config)
   # noise_pred = model(x_noisy, t)

    posterior_variance_t = get_index_from_list(constant_dict['posterior_variance'], t, noise_pred.shape, config)
    x_prime_noisy = x_noisy -  torch.sqrt(posterior_variance_t) * noise_pred
    x_noisy_for = x_noisy - torch.sqrt(posterior_variance_t) * noise

    feature_extractor = Feature_extractor(config, out_indices=[2])
    feature_extractor.to(config.model.device)
    F_x_noisy = feature_extractor(x_noisy_for.to(config.model.device))
    F_x_prime_noisy = feature_extractor(x_prime_noisy.to(config.model.device))
    for item in range(len(F_x_noisy)):
        cosloss += torch.mean(1-cos_loss(F_x_noisy[item].view(F_x_noisy[item].shape[0],-1),
                                      F_x_prime_noisy[item].view(F_x_prime_noisy[item].shape[0],-1)))

    return 0.5*(cosloss) + 0.5*(loss)



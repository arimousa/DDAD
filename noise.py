import torch
import numpy as np
import opensimplex
import random
from perlin_numpy import generate_fractal_noise_3d


def get_noise(x, config):
    if config.model.noise == 'Gaussian':
     #   torch.manual_seed(0)
        noise = torch.randn_like(x)
        return noise



    elif config.model.noise == 'Perlin': 
        #https://github.com/pvigier/perlin-numpy.git 
        # noise = perlin_noise()
        # final_noise = noise.unsqueeze(0).to(config.model.device)
        # for _ in range (x.shape[0]-1):
        #     noise =  perlin_noise().unsqueeze(0).to(config.model.device)
        #     final_noise= torch.cat((final_noise,noise), 0)
        # return final_noise
        noise = generate_fractal_noise_3d(
            (32, 256, 256), (1, 4, 4), 4, tileable=(True, False, False)
        )
        noise = torch.from_numpy(noise).float()
        noise = noise.unsqueeze(1).to(config.model.device)
        return noise


    else:
        print('noise is not selected correctly. Default is Gaussian noise')
        noise = torch.randn_like(x)
        return noise

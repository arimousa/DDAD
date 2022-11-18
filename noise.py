import torch
import numpy as np
import opensimplex


def get_noise(x, config):
    if config.model.noise == 'Gaussian':
        return torch.randn_like(x)


    elif config.model.noise == 'Simplex':     
        rng = np.random.default_rng()
        ixr, iyr = rng.random(256)*1.5, rng.random(256)*1.5
        noiser = opensimplex.noise2array(ixr, iyr)
        noiser = torch.Tensor(noiser)

        ixg, iyg =  rng.random(256)*1.5, rng.random(256)*1.5
        noiseg = opensimplex.noise2array(ixg, iyg)
        noiseg = torch.Tensor(noiseg)

        ixb, iyb =  rng.random(256)*1.5, rng.random(256)*1.5
        noiseb = opensimplex.noise2array(ixb, iyb)
        noiseb = torch.Tensor(noiseb)

        noise = torch.stack((noiser, noiseg, noiseb))

        # print(noise.shape)

        # noise = opensimplex.noise3([0,255],[0,255],[0,255])

        noise = torch.Tensor(noise)
        noise = noise.to(config.model.device)
        return noise


    else:
        print('Noise is not set correctly. Default is Gaussian noise')
        return torch.randn_like(x)
    





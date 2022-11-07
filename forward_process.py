import torch
from utilities import *
import matplotlib.pyplot as plt
import random
import opensimplex
import matplotlib.pyplot as plt
from utilities import *
import numpy

device = "cuda" if torch.cuda.is_available() else "cpu"



def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """

    noise = torch.randn_like(x_0)


    # rng = numpy.random.default_rng()
    # ixr, iyr = rng.random(256)*1.5, rng.random(256)*1.5
    # noiser = opensimplex.noise2array(ixr, iyr)
    # noiser = torch.Tensor(noiser)

    # ixg, iyg =  rng.random(256)*1.5, rng.random(256)*1.5
    # noiseg = opensimplex.noise2array(ixg, iyg)
    # noiseg = torch.Tensor(noiseg)

    # ixb, iyb =  rng.random(256)*1.5, rng.random(256)*1.5
    # noiseb = opensimplex.noise2array(ixb, iyb)
    # noiseb = torch.Tensor(noiseb)

    # noise = torch.stack((noiser, noiseg, noiseb))

   # print(noise.shape)






    # noise = opensimplex.noise3([0,255],[0,255],[0,255])
    # print('noise : ',noise)
    # noise = torch.Tensor(noise)
                    # plt.figure(figsize=(11,11))
                    # plt.axis('off')
                    # plt.subplot(1, 1, 1)
                    # plt.imshow(noise)
                    # plt.title('clear image')
                    # plt.savefig('results/nosise.png')
    

   # print('noise.shape : ',noise.shape)


    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)



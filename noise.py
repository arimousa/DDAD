import torch
import numpy as np
import opensimplex
import random



def get_noise(x, t, config):
    if config.model.noise == 'Gaussian':
     #   torch.manual_seed(0)
        noise = torch.randn_like(x)
        return noise



    elif config.model.noise == 'Perlin': 
        #https://github.com/pvigier/perlin-numpy.git 
        noise = perlin_noise()
        final_noise = noise.unsqueeze(0).to(config.model.device)
        for _ in range (x.shape[0]-1):
            noise =  perlin_noise().unsqueeze(0).to(config.model.device)
            final_noise= torch.cat((final_noise,noise), 0)
        return final_noise


    else:
        print('noise is not selected correctly. Default is Gaussian noise')
        noise = torch.randn_like(x)
        return noise
        
def perlin_noise():
    noiser = generate_fractal_noise_2d((256, 256), (8, 8), octaves=6, persistence=0.8)
    noiseg = generate_fractal_noise_2d((256, 256), (8, 8), octaves=6, persistence=0.8)
    noiseb= generate_fractal_noise_2d((256, 256), (8, 8), octaves=6, persistence=0.8)
    noiseg = torch.tensor(noiser, dtype=(torch.float))
    noiseb = torch.tensor(noiseg, dtype=(torch.float))
    noiser = torch.tensor(noiseb, dtype=(torch.float))
    noise = torch.stack((noiser, noiseg, noiseb))
    return noise

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise
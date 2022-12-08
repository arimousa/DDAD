import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from forward_process import *
from dataset import *
from sample import *

from noise import *

def visualize(image, noisy_image, GT, anomaly_map, index, category) :
    for idx, img in enumerate(image):
        plt.figure(figsize=(11,11))
        plt.axis('off')
        plt.subplot(1, 4, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')


        plt.subplot(1, 4, 2)
        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        

       
        plt.subplot(1, 4, 3)
        plt.imshow(show_tensor_mask(GT[idx]))
        plt.title('ground truth')

        plt.subplot(1, 4, 4)
        plt.imshow(show_tensor_mask(anomaly_map[idx]))
        plt.title('result')
        plt.savefig('results/{}sample{}.png'.format(category,index+idx))
        plt.close()



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

def show_tensor_mask(image):
    reverse_transforms = transforms.Compose([
       # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
       # transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
     #   transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)
        

@torch.no_grad()
def sample_plot_image(model, trainloader, constant_dict, epoch, category, config):
    image = next(iter(trainloader))[0]
    # Sample noise
    trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)

    image = forward_diffusion_sample(image, trajectoy_steps, constant_dict, config)[0]
    num_images = 5
    trajectory_steps = config.model.trajectory_steps
    stepsize = int(trajectory_steps/num_images)
    
    
    plt.figure(figsize=(15,15))
    plt.axis('off')
    image_to_show =show_tensor_image(image)
    plt.subplot(1, num_images+1, int(trajectory_steps/stepsize)+1)
    plt.imshow(image_to_show)
    plt.title(trajectory_steps)
    for i in range(0,trajectory_steps-1)[::-1]:
        t = torch.full((1,), i, device=config.model.device, dtype=torch.long)
        image = sample_timestep(config, model, constant_dict, image, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images+1, int(i/stepsize)+1)
            image_to_show =show_tensor_image(image.detach().cpu())
            plt.imshow(image_to_show)
            plt.title(i)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('results/{}backward_process_after_{}_epochs.png'.format(category, epoch))

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from forward_process import *
from dataset import *
from sample import *

from noise import *

def visualalize_rgb(image1,image2,image):
    plt.figure(figsize=(11,11))
    plt.subplot(1, 5, 1).axis('off')
    plt.subplot(1, 5, 2).axis('off')
    plt.subplot(1, 5, 3).axis('off')
    plt.subplot(1, 5, 4).axis('off')
    plt.subplot(1, 5, 5).axis('off')
    plt.subplot(1, 5, 1)
    plt.imshow(show_tensor_image(image[:,0,:,:].unsqueeze(1)))
    plt.title('r')
    plt.subplot(1, 5, 2)
    plt.imshow(show_tensor_image(image[:,1,:,:].unsqueeze(1)))
    plt.title('g')
    plt.subplot(1, 5, 3)
    plt.imshow(show_tensor_image(image[:,2,:,:].unsqueeze(1)))
    plt.title('b')
    plt.subplot(1, 5, 4)
    plt.imshow(show_tensor_image(image1))
    plt.title('image1')
    plt.subplot(1, 5, 5)
    plt.imshow(show_tensor_image(image2))
    plt.title('image2')

    k = 0
    while os.path.exists('results/rgb{}.png'.format(k)):
        k += 1
    plt.savefig('results/rgb{}.png'.format(k))
    plt.close()

def visualalize_distance(output, target, i_d, f_d):
    plt.figure(figsize=(11,11))
    plt.subplot(1, 4, 1).axis('off')
    plt.subplot(1, 4, 2).axis('off')
    plt.subplot(1, 4, 3).axis('off')
    plt.subplot(1, 4, 4).axis('off')

    plt.subplot(1, 4, 1)
    plt.imshow(show_tensor_image(target))
    plt.title('input image')
    

    plt.subplot(1, 4, 2)
    plt.imshow(show_tensor_image(output))
    plt.title('reconstructed image')

    plt.subplot(1, 4, 3)
    plt.imshow(show_tensor_image(f_d))
    plt.title('feature')

    plt.subplot(1, 4, 4)
    plt.imshow(show_tensor_image(i_d))
    plt.title('pixel')
    k = 0
    while os.path.exists('results/heatmap{}.png'.format(k)):
        k += 1
    plt.savefig('results/heatmap{}.png'.format(k))
    plt.close()

def visualize_reconstructed(input, data,s):
    for index in range(2):
        fig, axs = plt.subplots(int(len(data)/5),6)
        row = 0
        col = 1
        axs[0,0].imshow(show_tensor_image(input[index,:,:,:]))
        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 0].get_yaxis().set_visible(False)
        axs[0,0].set_title('input')
        for i, img in enumerate(data):
            axs[row, col].imshow(show_tensor_image(img[index,:,:,:]))
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)
            axs[row, col].set_title(str(i))
            col += 1
            if col == 6:
                row += 1
                col = 0
        col = 6
        row = int(len(data)/5)
        remain = col * row - len(data) -1
        for j in range(remain):
            col -= 1
            axs[row-1, col].remove()
            axs[row-1, col].get_xaxis().set_visible(False)
            axs[row-1, col].get_yaxis().set_visible(False)
            
        
            
        plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
        k = 0

        while os.path.exists(f'results/reconstructed{k}{s}.png'):
            k += 1
        plt.savefig(f'results/reconstructed{k}{s}.png')
        plt.close()



def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category) :
    for idx, img in enumerate(image):
        plt.figure(figsize=(11,11))
        plt.subplot(1, 2, 1).axis('off')
        plt.subplot(1, 2, 2).axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')

        plt.subplot(1, 2, 2)

        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        plt.savefig('results/{}sample{}.png'.format(category,idx))
        plt.close()

        plt.figure(figsize=(11,11))
        plt.subplot(1, 3, 1).axis('off')
        plt.subplot(1, 3, 2).axis('off')
        plt.subplot(1, 3, 3).axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(show_tensor_mask(GT[idx]))
        plt.title('ground truth')


        plt.subplot(1, 3, 2)
        plt.imshow(show_tensor_mask(pred_mask[idx]))
        plt.title('good' if torch.max(pred_mask[idx]) == 0 else 'bad', color="g" if torch.max(pred_mask[idx]) == 0 else "r")

        plt.subplot(1, 3, 3)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        plt.title('heat map')
        plt.savefig('results/{}sample{}heatmap.png'.format(category,idx))
        plt.close()



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
     #   transforms.ToPILImage(),
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
    if category == None:
        category = 'empty'
    plt.savefig('results/{}backward_process_after_{}_epochs.png'.format(category, epoch))

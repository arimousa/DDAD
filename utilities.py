import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np


def linear_beta_schedule(timesteps, start=0.0001, end=0.0101):  #end=0.0005 Epoch 200 | step 000 Loss: 0.16750505566596985  || end=0.005 Epoch 300 | step 000 Loss: 0.10944684594869614 
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)
        
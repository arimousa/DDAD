import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


from utilities import *



def heat_map(outputs, targets, config):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1

    anomaly_map = torch.ones([outputs.shape[0], 1, config.data.image_size, config.data.image_size], device = config.model.device) #* (-1)

    distance_map = 1 - F.cosine_similarity(outputs.to(config.model.device), targets.to(config.model.device)).to(config.model.device)
    distance_map = torch.unsqueeze(distance_map, dim=1)

    distance_map = F.interpolate(distance_map , size = config.data.image_size, mode="bilinear", align_corners=True)

    anomaly_map *= distance_map

    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    
    return anomaly_map
    
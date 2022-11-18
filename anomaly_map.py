import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision

from utilities import *
from backbone import *




def heat_map(outputs, targets, config):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1
    anomaly_map = torch.zeros([outputs[0].shape[0], 3, int(config.data.image_size), int(config.data.image_size)], device = config.model.device)#distance_map.shape[0]


    for output, target in zip(outputs, targets):
      #  output = F.interpolate(output , size = int(config.data.image_size/8), mode="bilinear")
      #  target = F.interpolate(target , size = int(config.data.image_size/8), mode="bilinear")

        feature_extractor = Feature_extractor(config)
        feature_extractor.to(config.model.device)
        outputs_features = feature_extractor(output.to(config.model.device))
        targets_features = feature_extractor(target.to(config.model.device))

            
        distance_map = 1 - F.cosine_similarity(outputs_features.to(config.model.device), targets_features.to(config.model.device),dim=1).to(config.model.device)
        distance_map = torch.unsqueeze(distance_map, dim=1)

        distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="bilinear")
        

        anomaly_map += distance_map
        
       # anomaly_map += (output-target).square()*2 - 1

    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_map = torchvision.transforms.functional.rgb_to_grayscale(anomaly_map)
    
    return anomaly_map
    
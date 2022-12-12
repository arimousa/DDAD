import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision

from utilities import *
from backbone import *




def heat_map(outputs, targets, mean_train_dataset, config, v):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1
    anomaly_map = torch.zeros([outputs[0].shape[0], 3, int(config.data.image_size), int(config.data.image_size)], device = config.model.device)#distance_map.shape[0]


    for output, target in zip(outputs, targets):
      #  output = F.interpolate(output , size = int(config.data.image_size/8), mode="bilinear")
      #  target = F.interpolate(target , size = int(config.data.image_size/8), mode="bilinear")

        feature_extractor = Feature_extractor(config, out_indices=[1])
        feature_extractor.to(config.model.device)
        outputs_features = feature_extractor(output.to(config.model.device))
        targets_features = feature_extractor(target.to(config.model.device))

        print('outputs_features : ',outputs_features.shape)   #([32, 512, 32, 32])
        print('output', output.shape)                         #([32, 3, 256, 256])


            
        distance_map = 1 - F.cosine_similarity(outputs_features.to(config.model.device), targets_features.to(config.model.device),dim=1).to(config.model.device)
        distance_map = torch.unsqueeze(distance_map, dim=1)
        distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="bilinear")

        distance_map_image = 1 - F.cosine_similarity(outputs_features.to(config.model.device), mean_train_dataset,dim=1).to(config.model.device)
        distance_map_image = torch.unsqueeze(distance_map_image, dim=1)
        distance_map_image = F.interpolate(distance_map_image , size = int(config.data.image_size), mode="bilinear")
        
        print('distance_map_image : ',torch.mean(distance_map_image))
        print('distance_map : ',torch.mean(distance_map))
        anomaly_map += ((v/100)*distance_map + ((100-v)/100)*distance_map_image)
        #anomaly_map = distance_map + distance_map_image
       # anomaly_map += (output-target).square()*2 - 1

    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    # anomaly_map = anomaly_map - torch.min(anomaly_map)
    # anomaly_map = anomaly_map / torch.max(anomaly_map)
    anomaly_map = torchvision.transforms.functional.rgb_to_grayscale(anomaly_map)
    #print('anomaly_map : ',anomaly_map.shape)
    
    return anomaly_map
    
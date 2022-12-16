import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision

from utilities import *
from backbone import *
from dataset import *




def heat_map(outputs, targets, config, v ,mean_train_dataset, representation_backbone):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1
    anomaly_score = torch.zeros([outputs[0].shape[0], 3, int(config.data.image_size), int(config.data.image_size)], device = config.model.device)#distance_map.shape[0]
    
    distance_map_image = representation_score(outputs[-1], config, mean_train_dataset, representation_backbone)
    
    for output in outputs:
        
      feature_extractor = Feature_extractor(config = config, backbone = "wide_resnet50_2", out_indices=[1])
      feature_extractor.to(config.model.device)
      outputs_features = feature_extractor(output.to(config.model.device))
      targets_features = feature_extractor(targets.to(config.model.device))
      
      distance_map = 1 - F.cosine_similarity(outputs_features.to(config.model.device), targets_features.to(config.model.device),dim=1).to(config.model.device)
      distance_map = torch.unsqueeze(distance_map, dim=1)
      distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="bilinear")
      anomaly_score += distance_map 

    print('distance_map_image : ',torch.mean(distance_map_image))
    print('distance_map : ',torch.mean(anomaly_score))

    anomaly_score = ((100-v)/100)*distance_map_image + ((v)/100)* anomaly_score



    anomaly_score = gaussian_blur2d(
        anomaly_score , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )

    anomaly_score = torchvision.transforms.functional.rgb_to_grayscale(anomaly_score)
    print( 'anomaly_score : ',torch.mean(anomaly_score))
    
    return anomaly_score


def representation_score(outputs, config, mean_train_dataset, representation_backbone):
    if representation_backbone == 'cait_m48_448':
      feature_extractor = Feature_extractor(config = config, backbone = "cait_m48_448", out_indices=[1])
      outputs = F.interpolate(outputs , size = 448, mode="bilinear")
    else :
      feature_extractor = Feature_extractor(config = config, backbone = "wide_resnet50_2", out_indices=[1])
    feature_extractor.to(config.model.device)
    outputs_features = feature_extractor(outputs.to(config.model.device))
    mean_train_dataset = mean_train_dataset.to(config.model.device)
    distance_map_image = 1 - F.cosine_similarity(outputs_features.to(config.model.device), mean_train_dataset,dim=1).to(config.model.device)
    distance_map_image = torch.unsqueeze(distance_map_image, dim=1)
    distance_map_image = F.interpolate(distance_map_image , size = int(config.data.image_size), mode="bilinear")
    return distance_map_image

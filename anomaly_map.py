import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision

from utilities import *
from backbone import *
from dataset import *
from visualize import *
from feature_extractor import *



def heat_map(outputs, targets, feature_extractor, constants_dict, config):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1
    
    i_d = color_distance(outputs, targets, config)
    f_d = feature_distance(outputs, targets,feature_extractor, constants_dict, config)


    print('image_distance : ',torch.mean(i_d))
    print('feature_distance : ',torch.mean(f_d))

    anomaly_score = (0.8) * f_d + (0.2) * i_d

    anomaly_score = gaussian_blur2d(
        anomaly_score , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_score = torch.sum(anomaly_score, dim=1).unsqueeze(1)
    print( 'anomaly_score : ',torch.mean(anomaly_score))
    
    return anomaly_score



def color_distance(image1, image2, config):
  image1 = (image1 - image1.min())/ (image1.max() - image1.min())
  image2 = (image2 - image2.min())/ (image2.max() - image2.min())

  # distance_map = 1 - F.cosine_similarity(image1.to(config.model.device), image2.to(config.model.device), dim=1).to(config.model.device)
  # distance_map = torch.unsqueeze(distance_map, dim=1)

  distance_map = image1.to(config.model.device) - image2.to(config.model.device) 
  distance_map = torch.abs(distance_map)

  # distance_map **= 2
  distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)
  return distance_map



def feature_distance(output, target,feature_extractor, constants_dict, config):

  outputs_features = extract_features(feature_extractor=feature_extractor, x=output.to(config.model.device), out_indices=[2,3], config=config) #feature_extractor(output.to(config.model.device))
  targets_features = extract_features(feature_extractor=feature_extractor, x=target.to(config.model.device), out_indices=[2,3], config=config) #feature_extractor(target.to(config.model.device))

  # outputs_features = (outputs_features - outputs_features.min())/ (outputs_features.max() - outputs_features.min())
  # targets_features = (targets_features - targets_features.min())/ (targets_features.max() - targets_features.min())
  
  # distance_map = (outputs_features.to(config.model.device) - targets_features.to(config.model.device))
  # distance_map = torch.abs(distance_map)
  # distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)

  distance_map = 1 - F.cosine_similarity(outputs_features.to(config.model.device), targets_features.to(config.model.device), dim=1).to(config.model.device)
  distance_map = torch.unsqueeze(distance_map, dim=1)

  distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="nearest")
  return distance_map


import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision

from utilities import *
from backbone import *
from dataset import *
from visualize import *
from feature_extractor import *
from PIL import Image



def heat_map(outputs, targets, feature_extractor, constants_dict, config):
    sigma = 4
    kernel_size = 2*int(4 * sigma + 0.5) +1
    anomaly_score = 0
    for output, target in zip(outputs, targets):
      i_d = color_distance(output, target, config)
      f_d = feature_distance(output, target,feature_extractor, constants_dict, config)
      print('image_distance : ',torch.mean(i_d))
      print('feature_distance : ',torch.mean(f_d))
      
      visualalize_distance(output, target, i_d, f_d)

      anomaly_score += i_d #(f_d + .4 * i_d)

    anomaly_score = gaussian_blur2d(
        anomaly_score , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_score = torch.sum(anomaly_score, dim=1).unsqueeze(1)
    print( 'anomaly_score : ',torch.mean(anomaly_score))
    
    return anomaly_score

def rgb_to_cmyk(r, g, b):
    RGB_SCALE = 1
    CMYK_SCALE = 100
    # if (r, g, b) == (0, 0, 0):
    #     # black
    #     return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = torch.zeros(c.shape, device=c.device)

    c = c.view(-1)
    m = m.view(-1)
    y = y.view(-1)
    min_cmy = min_cmy.view(-1)
    for i in range(len(c)):
      min_cmy[i] = min(c[i], m[i], y[i])
    c = c.view((256,256))
    m = m.view((256,256))
    y = y.view((256,256))
    min_cmy = min_cmy.view((256,256))
    
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE


def color_distance(image1, image2, config):
    image1 = ((image1 - image1.min())/ (image1.max() - image1.min())) 
    image2 = ((image2 - image2.min())/ (image2.max() - image2.min()))
    
    for i, (img1, img2) in enumerate(zip(image1, image2)):
        c1,m1,y1,k1 = rgb_to_cmyk(img1[0,:,:], img1[1,:,:], img1[2,:,:])
        c2,m2,y2,k2 = rgb_to_cmyk(img2[0], img2[1], img2[2])
        img1_cmyk = torch.stack((c1,m1,y1), dim=0)
        print('img1_cmyk : ',img1_cmyk.shape)
        img2_cmyk = torch.stack((c2,m2,y2), dim=0)
        img1_cmyk = img1_cmyk.to(config.model.device)
        img2_cmyk = img2_cmyk.to(config.model.device)
        distance_map = torch.abs(img1_cmyk - img2_cmyk).to(config.model.device)
        distance_map = torch.mean(distance_map, dim=0).unsqueeze(0)
        if i == 0:
          batch = distance_map
        else:
          batch = torch.cat((batch , distance_map) , dim=0)
    batch = batch.unsqueeze(1)
    print('batch :', batch.shape)
    return batch



    # distance_map = image1.to(config.model.device) - image2.to(config.model.device) 
    # distance_map = torch.abs(distance_map)
    # #visualalize_rgb(image1, image2 ,distance_map)

    # distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)
    # return distance_map



def feature_distance(output, target,feature_extractor, constants_dict, config):

  outputs_features = extract_features(feature_extractor=feature_extractor, x=output.to(config.model.device), out_indices=[2,3], config=config) #feature_extractor(output.to(config.model.device))
  targets_features = extract_features(feature_extractor=feature_extractor, x=target.to(config.model.device), out_indices=[2,3], config=config) #feature_extractor(target.to(config.model.device))

  outputs_features = (outputs_features - outputs_features.min())/ (outputs_features.max() - outputs_features.min())
  targets_features = (targets_features - targets_features.min())/ (targets_features.max() - targets_features.min())
  
  # distance_map = (outputs_features.to(config.model.device) - targets_features.to(config.model.device))
  # distance_map = torch.abs(distance_map)
  # distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)

  distance_map = 1 - F.cosine_similarity(outputs_features.to(config.model.device), targets_features.to(config.model.device), dim=1).to(config.model.device)
  distance_map = torch.unsqueeze(distance_map, dim=1)

  distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="nearest")

  # distance_map = torch.mean(torch.pow((outputs_features - targets_features), 2), dim=1).unsqueeze(1)

  return distance_map


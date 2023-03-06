import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision
from torchvision.transforms import transforms
import math 
from utilities import *
from backbone import *
from dataset import *
from visualize import *
from feature_extractor import *
# import cv2
import numpy as np


def heat_map(output, target, SFE, TFE, constants_dict, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    anomaly_map = 0


    output = output.to(config.model.device)
    target = target.to(config.model.device)

    
        

    i_d = color_distance(output, target, config) #torch.sqrt(torch.sum(((output)-(target))**2,dim=1).unsqueeze(1)) # 1 - F.cosine_similarity(patchify(output) , patchify(target), dim=1).to(config.model.device).unsqueeze(1) # color_distance(output, target, config) #torch.sqrt(torch.mean(((output)-(target))**2,dim=1).unsqueeze(1))   #torch.sqrt(torch.sum(((output)-(target))**2,dim=1).unsqueeze(1)) #color_distance(output, target, config)        ((output)-(target))**2  #torch.mean(torch.abs((output)-(target)),dim=1).unsqueeze(1)
    f_d = feature_distance((output),  (target), SFE, TFE, constants_dict, config)
    f_d = torch.Tensor(f_d).to(config.model.device)
    # print('image_distance mean : ',torch.mean(i_d))
    # print('feature_distance mean : ',torch.mean(f_d))
    # print('image_distance max : ',torch.max(i_d))
    # print('feature_distance max : ',torch.max(f_d))
    # i_d = torch.clamp(i_d, max=f_d.max().item())

    
    # visualalize_distance(output, target, i_d, f_d)

    anomaly_map += f_d #0.1 * i_d + 2 * f_d    # 2 for W5, 4 for W101

    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    # print( 'anomaly_map : ',torch.mean(anomaly_map))

    return anomaly_map

def rgb_to_cmyk(images):
    RGB_SCALE = 1
    CMYK_SCALE = 100

    cmy = 1 - images / RGB_SCALE

    min_cmy = torch.zeros(images.shape, device=images.device)
    min_cmy = torch.amin(cmy, dim=1).unsqueeze(1)-.001
    cmy = (cmy - min_cmy) / (1 - min_cmy)
    k = min_cmy
    cmyk = torch.cat((cmy,k), dim=1)
    return cmyk * CMYK_SCALE


def color_distance(image1, image2, config):
    # image1 = ((image1 - image1.min())/ (image1.max() - image1.min())) 
    # image2 = ((image2 - image2.min())/ (image2.max() - image2.min()))
    # image1_1 = (image1 + 1.0)/2.0
    # image2_2 = (image2 + 1.0)/2.0
  
    if config.model.backbone == 'deit_base_distilled_patch16_384':
        transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif config.model.backbone == 'cait_m48_448':
        transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(224), 
            # transforms.Lambda(lambda t: (t + 1) / (2)),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    image1 = transform(image1)
    image2 = transform(image2)
    # visualalize_distance(image1_1, image2_2, image1, image2)

    cmyk_image_1 = rgb_to_cmyk(image1)
    cmyk_image_2 = rgb_to_cmyk(image2)

    # cmyk_image_1 = ((cmyk_image_1 - cmyk_image_1.min())/ (cmyk_image_1.max() - cmyk_image_1.min())).to(config.model.device)
    # cmyk_image_2 = ((cmyk_image_2 - cmyk_image_2.min())/ (cmyk_image_2.max() - cmyk_image_2.min())).to(config.model.device)

    # distance_map = 1 - F.cosine_similarity(cmyk_image_1, cmyk_image_2, dim=1).to(config.model.device).unsqueeze(1)

    # distance_map = (cmyk_image_1 - cmyk_image_2)**2
    # distance_map = 1 - F.cosine_similarity((image1) , (image2), dim=1).to(config.model.device).unsqueeze(1)
    # distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)
    distance_map = torch.mean(((image1) - (image2))**2, dim=1).unsqueeze(1)
    distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="bilinear")
    return distance_map


def cal_anomaly_map(fs_list, ft_list, config, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = torch.ones([fs_list[0].shape[0], 1, out_size, out_size]).to(config.model.device)
    else:
        anomaly_map = torch.zeros([fs_list[0].shape[0] ,1 ,out_size, out_size]).to(config.model.device)
    a_map_list = []
    for i in range(len(ft_list)):
        if i == 0:
            continue
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(patchify(fs), patchify(ft))
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def feature_distance(output, target,SFE, TFE, constants_dict, config):

    # output = ((output - output.min())/ (output.max() - output.min())) 
    # target = ((target - target.min())/ (target.max() - target.min()))
    # output = (output + 1.0)/2.0
    # target = (target + 1.0)/2.0
    if config.model.backbone == 'deit_base_distilled_patch16_384':
        transform = transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.Resize((384,384)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif config.model.backbone == 'cait_m48_448':
        transform = transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.Resize((448,448)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(224), 
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    output = transform(output)
    target = transform(target)
    outpu_features = SFE(target)
    target_features = TFE(target)
    distance_map = cal_anomaly_map(outpu_features, target_features, config, out_size=256, amap_mode='a')[0]
    return distance_map 

    # outputs_features = extract_features(feature_extractor=feature_extractor, x=output, config=config, out_indices=['layer1']) 
    # targets_features = extract_features(feature_extractor=feature_extractor, x=target, config=config, out_indices=['layer1']) 

    # # p_id = patchify(outputs_features) - patchify(targets_features)

    # cosine_distance = 1 - F.cosine_similarity(patchify(outputs_features) , patchify(targets_features), dim=1).to(config.model.device).unsqueeze(1)

    # outputs_features2 = extract_features(feature_extractor=feature_extractor, x=output, config=config, out_indices=['layer2']) 
    # targets_features2 = extract_features(feature_extractor=feature_extractor, x=target, config=config, out_indices=['layer2']) 


    # cosine_distance2 = 1 - F.cosine_similarity((outputs_features2) , (targets_features2), dim=1).to(config.model.device).unsqueeze(1)


    # outputs_features3 = extract_features(feature_extractor=feature_extractor, x=output, config=config, out_indices=['layer3']) 
    # targets_features3 = extract_features(feature_extractor=feature_extractor, x=target, config=config, out_indices=['layer3']) 


    # cosine_distance3 = 1 - F.cosine_similarity((outputs_features3) , (targets_features3), dim=1).to(config.model.device).unsqueeze(1)



    # distance_map = F.interpolate(cosine_distance , size = int(config.data.image_size), mode="bilinear")
    # distance_map2 = F.interpolate(cosine_distance2 , size = int(config.data.image_size), mode="bilinear")
    # distance_map3 = F.interpolate(cosine_distance3 , size = int(config.data.image_size), mode="bilinear")


    # distance_map_t = distance_map #( distance_map +  distance_map2 +  distance_map3)  /3


    # return distance_map_t



def patchify(features, return_spatial_info=False):
    """Convert a tensor into a tensor of respective patches.
    Args:
        x: [torch.Tensor, bs x c x w x h]
    Returns:
        x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
        patchsize]
    """
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features


def unpatch_scores(x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])
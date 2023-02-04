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


def heat_map(outputs, targets, feature_extractor, constants_dict, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    anomaly_score = 0
    for output, target in zip(outputs, targets):
        output = output.to(config.model.device)
        target = target.to(config.model.device)

        i_d = color_distance(output, target, config)
        f_d = feature_distance(output, target, feature_extractor, constants_dict, config)
        # print('image_distance mean : ',torch.mean(i_d))
        # print('feature_distance mean : ',torch.mean(f_d))
        # print('image_distance max : ',torch.max(i_d))
        # print('feature_distance max : ',torch.max(f_d))
        
        visualalize_distance(output, target, i_d, f_d)

        anomaly_score += f_d + .8 * i_d #0.7 * f_d  + 0.3 * i_d # .8*

    anomaly_score = gaussian_blur2d(
        anomaly_score , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_score = torch.sum(anomaly_score, dim=1).unsqueeze(1)
    # print( 'anomaly_score : ',torch.mean(anomaly_score))

    return anomaly_score

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
    image1 = ((image1 - image1.min())/ (image1.max() - image1.min())) 
    image2 = ((image2 - image2.min())/ (image2.max() - image2.min())) 

    cmyk_image_1 = rgb_to_cmyk(image1)
    cmyk_image_2 = rgb_to_cmyk(image2)

    cmyk_image_1 = ((cmyk_image_1 - cmyk_image_1.min())/ (cmyk_image_1.max() - cmyk_image_1.min())).to(config.model.device)
    cmyk_image_2 = ((cmyk_image_2 - cmyk_image_2.min())/ (cmyk_image_2.max() - cmyk_image_2.min())).to(config.model.device)

    # distance_map = 1 - F.cosine_similarity(cmyk_image_1, cmyk_image_2, dim=1).to(config.model.device).unsqueeze(1)

    distance_map = (cmyk_image_1 - cmyk_image_2)
    distance_map = torch.abs(distance_map)
    distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)


    return distance_map



def feature_distance(output, target,feature_extractor, constants_dict, config):

    # output = ((output - output.min())/ (output.max() - output.min())) 
    # target = ((target - target.min())/ (target.max() - target.min())) 
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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    output = transform(output)
    target = transform(target)

   # print('output : ', output.max(), output.min())

    outputs_features = extract_features(feature_extractor=feature_extractor, x=output, out_indices=[2,3], config=config) 
    targets_features = extract_features(feature_extractor=feature_extractor, x=target, out_indices=[2,3], config=config) 

    # p_id = patchify(outputs_features) - patchify(targets_features)

    # cosine_distance = 1 - F.cosine_similarity(patchify(outputs_features) , patchify(targets_features), dim=1).to(config.model.device).unsqueeze(1)
    cosine_distance = 1 - F.cosine_similarity(outputs_features , targets_features, dim=1).to(config.model.device).unsqueeze(1)

    #euclidian_distance = torch.sqrt(torch.sum((outputs_features - targets_features)**2, dim=1).unsqueeze(1))
    # L1d = torch.sqrt(torch.sum((outputs_features - targets_features), dim=1).unsqueeze(1))
    # euclidian_distance = torch.cdist(outputs_features, targets_features, p=2)
    # print('euclidian_distance : ', euclidian_distance.shape)
    distance_map = F.interpolate(cosine_distance , size = int(config.data.image_size), mode="bilinear")


    return distance_map



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
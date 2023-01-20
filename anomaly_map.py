import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision
from torchvision.transforms import transforms
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
        i_d = color_distance(output, target, config)
        f_d = feature_distance(output, target, feature_extractor, constants_dict, config)
        # print('image_distance mean : ',torch.mean(i_d))
        # print('feature_distance mean : ',torch.mean(f_d))
        # print('image_distance max : ',torch.max(i_d))
        # print('feature_distance max : ',torch.max(f_d))
        
        visualalize_distance(output, target, i_d, f_d)

        anomaly_score += f_d  + .8* i_d #0.7 * f_d  + 0.3 * i_d # .8*

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

    cmyk_image_1 = ((cmyk_image_1 - cmyk_image_1.min())/ (cmyk_image_1.max() - cmyk_image_1.min())) 
    cmyk_image_2 = ((cmyk_image_2 - cmyk_image_2.min())/ (cmyk_image_2.max() - cmyk_image_2.min()))

    distance_map = (cmyk_image_1.to(config.model.device) - cmyk_image_2.to(config.model.device) )
    
   # distance_map = torch.sqrt(torch.sum(distance_map, dim=1).unsqueeze(1))
    distance_map = torch.mean(distance_map, dim=1).unsqueeze(1)
    return distance_map



def feature_distance(output, target,feature_extractor, constants_dict, config):

    # output = ((output - output.min())/ (output.max() - output.min())) 
    # target = ((target - target.min())/ (target.max() - target.min())) 

    # reversed = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # output = reversed(output)
    # target = reversed(target)

   # print('output : ', output.max(), output.min())

    outputs_features = extract_features(feature_extractor=feature_extractor, x=output.to(config.model.device), out_indices=[2,3], config=config) 
    targets_features = extract_features(feature_extractor=feature_extractor, x=target.to(config.model.device), out_indices=[2,3], config=config) 

    outputs_features =( (outputs_features - outputs_features.min())/ (outputs_features.max() - outputs_features.min()) ).to(config.model.device)
    targets_features = ((targets_features - targets_features.min())/ (targets_features.max() - targets_features.min())).to(config.model.device)


    cosine_distance = 1 - F.cosine_similarity(outputs_features, targets_features, dim=1).to(config.model.device).unsqueeze(1)
    #euclidian_distance = torch.sqrt(torch.sum((outputs_features - targets_features)**2, dim=1).unsqueeze(1))
    # euclidian_distance = torch.cdist(outputs_features, targets_features, p=2)
    # print('euclidian_distance : ', euclidian_distance.shape)
    distance_map = F.interpolate(cosine_distance , size = int(config.data.image_size), mode="bilinear")


    return distance_map


    # patches1_features = []
    # patches2_features = []
    # patch_size = (64, 64)
    # stride = (32, 32)
    # print('output : ', output.shape)
    # # patchify the two images
    # patches1 = output.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])
    # patches2 = target.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])
    # print('patches1 : ', len(patches1))
    # print('patches[0] : ', patches1[0].shape)

    # patches1 = torch.stack(patches1, dim=0)
    # patches2 = torch.stack(patches2, dim=0)
    # print('patches1 stack : ', patches1.shape)
    # for patch1, patch2 in zip(patches1, patches2):
    #     patch1_feature = extract_features(feature_extractor=feature_extractor, x=patch1.to(config.model.device), out_indices=[2,3], config=config) 
    #     patch2_feature = extract_features(feature_extractor=feature_extractor, x=patch2.to(config.model.device), out_indices=[2,3], config=config) 
    #     patches1_features.append(patch1_feature)
    #     patches2_features.append(patch2_feature)

    
    # print('image1 : ', image1.shape)

    # image1 = image1.view(32,3,256,256)
    # image2 = image2.view(32,3,256,256)
    # distance_map = 1 - F.cosine_similarity(image1.to(config.model.device), image2.to(config.model.device), dim=1).to(config.model.device)
    # distance_map = torch.unsqueeze(distance_map, dim=1)
    # distance_map = F.interpolate(distance_map , size = int(config.data.image_size), mode="bilinear")

# def patchify(img, patch_size):
#     patches = []
#     for i in range(0, img.shape[1], patch_size):
#         for j in range(0, img.shape[2], patch_size):
#             patch = img[:, i:i+patch_size, j:j+patch_size]
#             patches.append(patch)
#     return patches
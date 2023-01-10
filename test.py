from asyncio import constants
import torch
from model import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from backbone import *
from metrics import metric
from feature_extractor import *

from EMA import EMAHelper
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def validate(model, constants_dict, config):

    test_dataset = Dataset(
        root= config.data.data_dir,
        category=config.data.category,
        config = config,
        is_train=False,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= config.data.batch_size,
        shuffle=False,
        num_workers= config.model.num_workers,
        drop_last=False,
    )


    
    feature_extractor = tune_feature_extractor(constants_dict, config)

    labels_list = []
    predictions= []
    anomaly_map_list = []
    GT_list = []
    reconstructed_list = []
    forward_list = []
    for data, targets, labels in testloader:    
        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)
        noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0]
        seq = range(0, config.model.test_trajectoy_steps, config.model.skip)
        H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
        data = data.to(config.model.device)
        noisy_image = noisy_image.to(config.model.device)
        reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, cls_fn=None, classes=None) 
        #reconstructed, _ = generalized_steps(noisy_image, seq, model,  constants_dict['betas'], config)
        data_reconstructed = reconstructed[-1]

        forward_list_compare = [data]
        reconstructed_compare = [data_reconstructed]

        # forward_list_compare.append(forward_diffusion_sample(data, torch.Tensor([4 * config.model.skip]).type(torch.int64), constants_dict, config)[0])
        # reconstructed_compare.append(reconstructed[-5])

        # forward_list_compare.append(forward_diffusion_sample(data, torch.Tensor([9 * config.model.skip]).type(torch.int64), constants_dict, config)[0])
        # reconstructed_compare.append(reconstructed[-10])

        visulalize_reconstructed(data, reconstructed)

        

        anomaly_map = heat_map(reconstructed_compare, forward_list_compare, feature_extractor, constants_dict, config)
        
        forward_list.append(data)
        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
        reconstructed_list.append(data_reconstructed)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(0 if label == 'good' else 1)
            predictions.append( torch.max(pred).item())

    
    threshold = metric(labels_list, predictions, anomaly_map_list, GT_list, config)

    forward_list = torch.cat(forward_list, dim=0)
    reconstructed_list = torch.cat(reconstructed_list, dim=0)
    anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
    pred_mask = (anomaly_map_list>threshold).float()
    GT_list = torch.cat(GT_list, dim=0)
    visualize(forward_list, reconstructed_list, GT_list, pred_mask, anomaly_map_list, config.data.category)
    


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


    # mean_train_dataset = torch.zeros([config.data.imput_channel, config.data.image_size, config.data.image_size]).to(config.model.device)
    # n_saples = 0
    # for step, batch in enumerate(trainloader):
    #     batch = torch.Tensor(batch[0])
    #     batch_mean = torch.mean(batch, dim=0)
    #     batch_mean = batch_mean.to(config.model.device)
    #     mean_train_dataset += batch_mean
    #     n_saples += batch[0].shape[0]
    # mean_train_dataset /= n_saples

    
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
        reconstructed, _ = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, cls_fn=None, classes=None) #generalized_steps(noisy_image, seq, model, constants_dict['betas'], config, eta=config.model.eta)
        data_reconstructed = reconstructed[-1]
        forward_list.append(data)

        anomaly_map = heat_map(data_reconstructed, data, feature_extractor, constants_dict, config)
        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
        reconstructed_list.append(reconstructed[-1])

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
    


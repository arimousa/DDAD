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

from EMA import EMAHelper
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def validate(model, constants_dict, config, category, v):

    test_dataset = MVTecDataset(
        root= config.data.data_dir,
        category=category,
        input_size=config.data.image_size,
        is_train=False,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= config.data.batch_size,
        shuffle=False,
        num_workers= config.model.num_workers,
        drop_last=False,
    )

    train_dataset = MVTecDataset(
        root= config.data.data_dir,
        category=category,
        input_size= config.data.image_size,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )

    # mean_train_dataset = torch.zeros([3 ,config.data.image_size, config.data.image_size])
    mean_train_dataset = torch.zeros([256, 64, 64]).to(config.model.device)
    n_saples = 0
    feature_extractor = Feature_extractor(config, out_indices=[1])
    feature_extractor.to(config.model.device)
    for step, batch in enumerate(trainloader):
        batch_features = feature_extractor(batch[0].to(config.model.device))
        print('batch_features : ',batch_features.shape)
        batch_features = torch.mean(batch_features, dim=0)
        batch_features = batch_features.to(config.model.device)
        mean_train_dataset += batch_features
       # mean_train_dataset += batch[0].sum(dim=0)
        n_saples += batch[0].shape[0]
    mean_train_dataset /= n_saples
    mean_train_dataset = mean_train_dataset.to(config.model.device)

 
    labels_list = []
    predictions_max = []
    predictions_mean = []
    anomaly_map_list = []
    GT_list = []
    index = 0
    for data, targets, labels in testloader:
        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)
        

        data_forward = []
        data_reconstructed = []

        
        noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0]
        for i in range(0,test_trajectoy_steps)[::-1]:
            t = torch.full((1,), i, device=config.model.device, dtype=torch.long)
            noisy_image = sample_timestep(config, model, constants_dict,  noisy_image.to(config.model.device), t)
            if i in  [0,5,10,15,20]:
                f_image = forward_diffusion_sample(data, t , constants_dict, config)[0]
                data_forward.append(f_image)
                data_reconstructed.append(noisy_image)
        
            

        
        anomaly_map = heat_map(data_reconstructed, data_forward, mean_train_dataset , config, v)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(0 if label == 'good' else 1)
            predictions_max.append( torch.max(pred).item())
            predictions_mean.append( torch.mean(pred).item())


        visualize(data, noisy_image, targets, anomaly_map, index, category) 
        index = index + data.shape[0] 

        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
    
    metric(labels_list, predictions_max, predictions_mean, anomaly_map_list, GT_list, config)

    
    
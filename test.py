from asyncio import constants
import torch
from unet import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from backbone import *
from metrics import *
from feature_extractor import *
import time
from datetime import timedelta

from EMA import EMAHelper
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure


def train_data_mean(testloader, feature_extractor, config):
    if config.data.name == 'MVTec':
        train_dataset = MVTecDataset(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=True,
        )
    num = 0
    for step, batch in enumerate(trainloader):
        data = batch[0].to(config.model.device)
        feature = extract_features(feature_extractor, data, config, out_indices=['layer1'])
        if step == 0:
            F_mean = (feature.sum(dim=0) / feature.shape[0])
        else:
            F_mean += (feature.sum(dim=0) / feature.shape[0])
        num += 1
    F_mean = F_mean / num
    return F_mean.unsqueeze(0)


def cr_function(x, train_mean, feature_extractor, config):
    x = x.to(config.model.device)
    x_feature = extract_features(feature_extractor, x, config, out_indices=['layer1'])

    cr = torch.sum(torch.abs(x_feature - train_mean),dim=1)
    cr = F.interpolate(cr.unsqueeze(1), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
    cr = cr.expand(x.shape[0], 3, x.shape[2], x.shape[3])
    cr = 1/((cr+1)*10)
    return cr
        



def validate(unet, constants_dict, config):

    if config.data.name == 'MVTec' or config.data.name == 'BTAD' or config.data.name == 'MTD' or config.data.name =='VisA_pytorch':
        test_dataset = MVTecDataset(
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

    
    if config.data.name == 'cifar10':
        trainloader, testloader = load_data(dataset_name='cifar10')
    SFE, TFE, bn = tune_feature_extractor(constants_dict, unet, config)
    # f_mean = train_data_mean(testloader, feature_extractor, config)

    labels_list = []
    predictions= []
    anomaly_map_list = []
    GT_list = []
    reconstructed_list = []
    forward_list = []
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.model.device)
    if config.data.name == 'cifar10':
        with torch.no_grad():
            for data, labels in testloader:
                data = data.to(config.model.device)
                    
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps2]).type(torch.int64).to(config.model.device)
                # noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                at = compute_alpha(constants_dict['betas'], test_trajectoy_steps.long(),config)
                noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')

                k = 0             

                seq = range(0 , config.model.test_trajectoy_steps2, config.model.skip2)

                cr = 0.2 # 0.2 for prediction interpolation
                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config,  eta2= 2 , eta3=1, constants_dict=constants_dict ,eraly_stop = False)
                data_reconstructed = reconstructed[-1]

                # visualize_reconstructed(data, reconstructed, k)

                anomaly_map = heat_map(reconstructed[-1], data, SFE, TFE, bn, constants_dict, config)

                # transform = transforms.Compose([
                #     transforms.CenterCrop((224)), 
                # ])
                # anomaly_map = transform(anomaly_map)
                targets = torch.zeros([data.shape[0],1, data.shape[-2], data.shape[-1]])
                # targets = transform(targets)
                
                forward_list.append(data)
                anomaly_map_list.append(anomaly_map)
                GT_list.append(targets)
                reconstructed_list.append(data_reconstructed)

        
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 0 else 1)
                    # predictions.append( 1 if torch.max(pred).item() > 0.1 else 0)
                    predictions.append(torch.max(pred).item() )



    # start = time.time()

    if config.data.name == 'MVTec' or config.data.name == 'BTAD' or config.data.name == 'MTD' or config.data.name =='VisA_pytorch':
        with torch.no_grad():
            for data, targets, labels in testloader:
                data = data.to(config.model.device)
                    
                test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps2]).type(torch.int64).to(config.model.device)
                # noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
                at = compute_alpha(constants_dict['betas'], test_trajectoy_steps.long(),config)
                noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')

                seq = range(0 , config.model.test_trajectoy_steps2, config.model.skip2)


                # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
                # reconstructed, rec_x0 = efficient_generalized_steps(config, constants_dict, noisy_image, seq, unet,  constants_dict['betas'], H_funcs, data, gama = .00,sigma_0 = 0.1, cls_fn=None, classes=None, early_stop=False)

                cr = 2 # 0.2 for prediction interpolation
                reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config, eta2= 4, eta3=0 , constants_dict=constants_dict ,eraly_stop = False)
                data_reconstructed = reconstructed[-1]


                anomaly_map = heat_map(data_reconstructed, data, SFE, TFE, bn, constants_dict, config)

                transform = transforms.Compose([
                    transforms.CenterCrop((224)), 
                ])
                anomaly_map = transform(anomaly_map)
                targets = transform(targets)

                
                forward_list.append(data)
                anomaly_map_list.append(anomaly_map)
                GT_list.append(targets)
                reconstructed_list.append(data_reconstructed)

        
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    # predictions.append( 1 if torch.max(pred).item() > 0.1 else 0)
                    predictions.append(torch.max(pred).item())
                    # predictions.append(torch.mean(torch.topk(pred,5)[0]).item())
                
                

        
    # end = time.time()
    # print('Inference time is ', str(timedelta(seconds=end - start)))

    
    threshold = metric(labels_list, predictions, anomaly_map_list, GT_list, config)
    print('threshold: ', threshold)

    

    reconstructed_list = torch.cat(reconstructed_list, dim=0)
    forward_list = torch.cat(forward_list, dim=0)

    
    
    anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
    pred_mask = (anomaly_map_list> threshold).float()
    GT_list = torch.cat(GT_list, dim=0)
    visualize(forward_list, reconstructed_list, GT_list, pred_mask, anomaly_map_list, config.data.category)
    
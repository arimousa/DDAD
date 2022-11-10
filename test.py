from asyncio import constants
import os
import glob
import torch
from model import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from PIL import Image
from torchmetrics import ROC, AUROC, F1Score
from anomaly_map import *
from backbone import *
from torchmetrics.classification import BinaryAUROC

from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def test_sample_timestep(config, model, x, t, constants_dict):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(constants_dict['betas'], t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        constants_dict['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(constants_dict['sqrt_recip_alphas'], t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(constants_dict['posterior_variance'], t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)

        # rng = numpy.random.default_rng()
        # ixr, iyr = rng.random(256)*1.5, rng.random(256)*1.5
        # noiser = opensimplex.noise2array(ixr, iyr)
        # noiser = torch.Tensor(noiser)

        # ixg, iyg =  rng.random(256)*1.5, rng.random(256)*1.5
        # noiseg = opensimplex.noise2array(ixg, iyg)
        # noiseg = torch.Tensor(noiseg)

        # ixb, iyb =  rng.random(256)*1.5, rng.random(256)*1.5
        # noiseb = opensimplex.noise2array(ixb, iyb)
        # noiseb = torch.Tensor(noiseb)

        # noise = torch.stack((noiser, noiseg, noiseb)).to(config.model.device)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 




@torch.no_grad()
def validate(model, constants_dict, config, category):

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
    

  #  writer = SummaryWriter('runs/DDAD')

   # for j in [0,10,20,30,40,50,60,70,80,90,100,120][::-1]: #0,10,20,30,40,

   # for tj in [100,150,200,250,300,350,400,450,499][::-1]:
    labels_list = []
    predictions_max = []
    predictions_mean = []
    index = 0
    for data, targets, labels in testloader:


    #  feature_extractor = Feature_extractor(config)
    #  feature_extractor.to(config.model.device)

    #  data_features = feature_extractor(data.to(config.model.device))


        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)
        noisy_image =  forward_diffusion_sample(data, test_trajectoy_steps, constants_dict['sqrt_alphas_cumprod'], 
                                            constants_dict['sqrt_one_minus_alphas_cumprod'], device=config.model.device)[0]
        
        

        data_forward = []
        data_reconstructed = []

        for i in range(0,test_trajectoy_steps)[::-1]:
            t = torch.full((1,), i, device=config.model.device, dtype=torch.long)
            noisy_image = test_sample_timestep(config, model, noisy_image.to(config.model.device), t, constants_dict)
            if i in  [0,10,20,30]: #,10,20,30,40,50,60
            #    anomaly_map = heat_map(noisy_image, data, config)
                f_image = forward_diffusion_sample(data, t , constants_dict['sqrt_alphas_cumprod'], constants_dict['sqrt_one_minus_alphas_cumprod'], device)[0]
                data_forward.append(f_image)
                data_reconstructed.append(noisy_image)
            

        
        anomaly_map = heat_map(data_reconstructed, data_forward, config)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(1 if label == 'good' else 0)
            predictions_max.append( torch.max(pred).item())
            predictions_mean.append( torch.mean(pred).item())

        anomaly_map.to(config.model.device)

        visualize(data, noisy_image, targets, anomaly_map, index, category)
        index = index + data.shape[0] 
    
    labels_list = torch.tensor(labels_list)
    predictions_max = torch.tensor(predictions_max)
    predictions_mean = torch.tensor(predictions_mean)

    
    roc = ROC()
    auroc = AUROC()
    fpr, tpr, thresholds = roc(predictions_max, labels_list)
    print('thresholds : ',thresholds)
    for index, (label, pred) in enumerate(zip(labels_list, predictions_max)):
        print('index:',index,' pred: ',pred,'  label: ',label)
    auroc_max = auroc(predictions_max, labels_list)
    auroc_mean = auroc(predictions_mean, labels_list)

    f1 = F1Score()
    f1_scor_max = f1(predictions_max, labels_list)
    f1_scor_mean = f1(predictions_mean, labels_list)

    metric = BinaryAUROC(thresholds=None)
    b_auroc = metric(predictions_max, labels_list)

    # writer.add_scalar('auroc', auroc, j)
    # writer.add_scalar('f1_scor', f1_scor, j)


    print("  AUROC_max: {}   |   AUROC_mean: {}   |  F1SCORE_max: {}    |  F1SCORE_mean: {}    |  Binary_AUROC: {}".format(auroc_max, auroc_mean, f1_scor_max, f1_scor_mean, b_auroc ))

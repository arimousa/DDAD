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

from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def test_sample_timestep(config, model, x, t, constants_dict):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(constants_dict['betas'], t, x.shape, config)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        constants_dict['sqrt_one_minus_alphas_cumprod'], t, x.shape, config
    )
    sqrt_recip_alphas_t = get_index_from_list(constants_dict['sqrt_recip_alphas'], t, x.shape, config)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(constants_dict['posterior_variance'], t, x.shape, config)
    
    if t == 0:
        return model_mean
    else:
        noise = get_noise(x, t, config)
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

        for j in range(0,10):
            noisy_image =  forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0]
            for i in range(0,test_trajectoy_steps)[::-1]:
                t = torch.full((1,), i, device=config.model.device, dtype=torch.long)
                noisy_image = test_sample_timestep(config, model, noisy_image.to(config.model.device), t, constants_dict)
                if j == 9:
                    if i in  [0,5,10]: 
                        f_image = forward_diffusion_sample(data, t , constants_dict, config)[0]
                        data_forward.append(f_image)
                        data_reconstructed.append(noisy_image)
            

        
        anomaly_map = heat_map(data_reconstructed, data_forward, config)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(0 if label == 'good' else 1)
            predictions_max.append( torch.max(pred).item())
            predictions_mean.append( torch.mean(pred).item())


        visualize(data, noisy_image, targets, anomaly_map, index, category) 
        index = index + data.shape[0] 

        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
    
    metric(labels_list, predictions_max, predictions_mean, anomaly_map_list, GT_list, config)

    
    
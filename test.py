from asyncio import constants
import torch
from unet import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *



def evaluate(unet, config):
    '''
    Evaluate the model on the test set
    visulaize the results
    '''
    test_dataset = Dataset_maker(
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

    
   
    feature_extractor = Domain_adaptation(unet, config, fine_tune=False)

    labels_list = []
    predictions= []
    anomaly_map_list = []
    gt_list = []
    reconstructed_list = []
    forward_list = []
   


    with torch.no_grad():
        for data, targets, labels in testloader:
            data = data.to(config.model.device)
                
            test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64).to(config.model.device)
            at = compute_alpha( test_trajectoy_steps.long(),config)
            noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
            seq = range(0 , config.model.test_trajectoy_steps, config.model.skip)
            reconstructed = Reconstruction(data, noisy_image, seq, unet, config, config.model.w)
            data_reconstructed = reconstructed[-1]


            anomaly_map = heat_map(data_reconstructed, data, feature_extractor, config)

            transform = transforms.Compose([
                transforms.CenterCrop((224)), 
            ])

            anomaly_map = transform(anomaly_map)
            targets = transform(targets)
            forward_list.append(data)
            anomaly_map_list.append(anomaly_map)
            gt_list.append(targets)
            reconstructed_list.append(data_reconstructed)
            for pred, label in zip(anomaly_map, labels):
                labels_list.append(0 if label == 'good' else 1)
                predictions.append(torch.max(pred).item())
                
                

    
    threshold = metric(labels_list, predictions, anomaly_map_list, gt_list, config)
    print('threshold: ', threshold)
    reconstructed_list = torch.cat(reconstructed_list, dim=0)
    forward_list = torch.cat(forward_list, dim=0)
    anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
    pred_mask = (anomaly_map_list> threshold).float()
    gt_list = torch.cat(gt_list, dim=0)
    if not os.path.exists('results'):
            os.mkdir('results')
    visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, config.data.category)
    



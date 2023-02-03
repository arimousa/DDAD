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
        data = data.to(config.model.device)
            
        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64).to(config.model.device)
        noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0].to(config.model.device)
        k = 0
        # while os.path.exists('results/Forward{}_1.png'.format(k)):
        #     k += 1
        # plt.figure(figsize=(11,11))
        # plt.subplot(1, 1, 1).axis('off')
        # plt.subplot(1, 1, 1)
        # plt.imshow(show_tensor_image(noisy_image))
        # plt.title('forward 1') 
        # plt.savefig('results/Forward{}_1.png'.format(k))



        seq = range(0 , config.model.test_trajectoy_steps, config.model.skip)
        # print('seq : ',seq)
        # H_funcs = Denoising(config.data.imput_channel, config.data.image_size, config.model.device)
        reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0.3, eraly_stop = True)
        # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .6, cls_fn=None, classes=None) 
        data_reconstructed = reconstructed[-1]

        plt.figure(figsize=(11,11))
        plt.subplot(1, 2, 1).axis('off')
        plt.subplot(1, 2, 2).axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(show_tensor_image(data_reconstructed))
        plt.title('Reconstructed 1') 
        plt.subplot(1, 2, 2)
        plt.imshow(show_tensor_image(rec_x0[-1]))
        plt.title('Guess 1') 
        plt.savefig('results/Reconstruct{}_1.png'.format(k))

        r = 6
        for i in range(r):
            noisy_image = forward_ti_steps(test_trajectoy_steps, 3 * config.model.skip, data_reconstructed, data, constants_dict['betas'], config)
            
            reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0.3, eraly_stop = True)
            data_reconstructed = reconstructed[-1]
            # j = 0
            # while os.path.exists('results/guess{}_1.png'.format(j)):
            #     j += 1
            # plt.figure(figsize=(11,11))
            # plt.subplot(1, 2, 1).axis('off')
            # plt.subplot(1, 2, 2).axis('off')
            # plt.subplot(1, 2, 1)
            # plt.imshow(show_tensor_image(rec_x0[-1]))
            # plt.title('Guess 1') 
            # plt.subplot(1, 2, 2)
            # plt.imshow(show_tensor_image(data_reconstructed))
            # plt.title('rec 1') 
            # plt.savefig('results/guess{}_1.png'.format(j))


        # reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0.3)
        # reconstructed, _ = generalized_steps(noisy_image, seq, model,  constants_dict['betas'], config)
        # data_reconstructed = reconstructed[-1]

        # visualize_reconstructed(data, rec_x0,s=1)
        # visualize_reconstructed(data, reconstructed,s=2)

        seq = range(0, config.model.test_trajectoy_steps2, config.model.skip2)
        noisy_image = forward_ti_steps(test_trajectoy_steps, 3 * config.model.skip, data_reconstructed, data, constants_dict['betas'], config)
        # plt.figure(figsize=(11,11))
        # plt.subplot(1, 1, 1).axis('off')
        # plt.subplot(1, 1, 1)
        # plt.imshow(show_tensor_image(noisy_image))
        # plt.title('forward 2') 
        # plt.savefig('results/Forward{}_for_last_step.png'.format(k))
        # reconstructed, rec_x0 = efficient_generalized_steps(config, noisy_image, seq, model,  constants_dict['betas'], H_funcs, data, gama = .4, cls_fn=None, classes=None) 
        reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, model, constants_dict['betas'], config, gama=0.1, eraly_stop = False)
        data_reconstructed = reconstructed[-1]

        plt.figure(figsize=(11,11))
        plt.subplot(1, 1, 1).axis('off')
        plt.subplot(1, 1, 1)
        plt.imshow(show_tensor_image(data_reconstructed))
        plt.title('Reconstructed') 
        plt.savefig('results/Reconstruct{}_last_reconstruction.png'.format(k))





        

        # visualize_reconstructed(data, rec_x0,s=3)
        # visualize_reconstructed(data, reconstructed,s=4)

        forward_list_compare = [] # [data]
        reconstructed_compare = [] # [data_reconstructed]

        forward_list_compare.append(data)
        reconstructed_compare.append(reconstructed[-1])

        # forward_list_compare.append(forward_diffusion_sample(data, torch.Tensor([9 * config.model.skip]).type(torch.int64), constants_dict, config)[0])
        # reconstructed_compare.append(reconstructed[-10])

        

        

        anomaly_map = heat_map(reconstructed_compare, forward_list_compare, feature_extractor, constants_dict, config)
        
        forward_list.append(data)
        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
        reconstructed_list.append(data_reconstructed)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(0 if label == 'good' else 1)
            # predictions.append( 1 if torch.max(pred).item() > 0.1 else 0)
            predictions.append(torch.max(pred).item() )

    
    threshold = metric(labels_list, predictions, anomaly_map_list, GT_list, config)
    print('threshold: ', threshold)

    forward_list = torch.cat(forward_list, dim=0)
    reconstructed_list = torch.cat(reconstructed_list, dim=0)
    anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
    pred_mask = (anomaly_map_list> threshold).float()
    GT_list = torch.cat(GT_list, dim=0)
    visualize(forward_list, reconstructed_list, GT_list, pred_mask, anomaly_map_list, config.data.category)
    

import torch
import os
import torch.nn as nn
from forward_process import *
from dataset import *

from torch.optim import Adam
from dataset import *
from backbone import *
from noise import *
from visualize import show_tensor_image

from torch.utils.tensorboard import SummaryWriter
from test import *
from loss import *
from optimizer import *
from sample import *

from EMA import EMAHelper





def trainer2(model, constants_dict, ema_helper, config):
    optimizer = build_optimizer(model, config) 
    trainloader = CIFAR10_dataset(config)


    writer = SummaryWriter('runs/DDAD')

    for epoch in range(config.model.epochs):
        for step, data in enumerate(trainloader):
            print('type data is : ',data)
            
            t = torch.randint(0, config.model.trajectory_steps, (data.data.shape[0],), device=config.model.device).long()


            optimizer.zero_grad()
            loss = get_loss(model, constants_dict, data.data, t, config) 
            loss.backward()
            optimizer.step()
            
            if config.model.ema:
                ema_helper.update(model)
            if epoch % 100 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch %1000 == 0 and epoch>0 and step ==0:
                # sample_plot_image(model, trainloader, constant_dict, epoch, category, config)
                if config.model.save_model:
                    if config.data.category:
                        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
                    else:
                        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch))) #config.model.checkpoint_name

            # if epoch %150 == 0  and step ==0: #and epoch>0
            #         validate(model, constants_dict, config, category, v_train)
            #     sample_plot_image(model, trainloader, constants_dict, epoch, category, config)
                
    if config.model.save_model:
        if config.data.category:
            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
        else:
            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1)))
   
    writer.flush()
    writer.close()
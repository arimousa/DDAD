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





def trainer(model, constants_dict, ema_helper, category, config):
        optimizer = build_optimizer(model, config)
        if config.data.name == 'MVTec' or config.data.name == 'BTAD' or config.data.name == 'MTD' or config.data.name == 'VisA_pytorch':
            train_dataset = MVTecDataset(
                root= config.data.data_dir,
                category=category,
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
        # if config.data.name == 'VisA':
        #     VisA()
        if config.data.name == 'cifar10':
            trainloader, testloader = load_data(dataset_name='cifar10')

        writer = SummaryWriter('runs/DDAD')
        
        for epoch in range(config.model.epochs):
            for step, batch in enumerate(trainloader):
                
                t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()


                optimizer.zero_grad()
                loss = get_loss(model, constants_dict, batch[0], t, config) 
                loss.backward()
                optimizer.step()
                
                if config.model.ema:
                    ema_helper.update(model)
                if epoch % 50 == 0 and step == 0:
                    print(f"Epoch {epoch} | Loss: {loss.item()}")
                if epoch %250 == 0 and step ==0:
                    # sample_plot_image(model, trainloader, constant_dict, epoch, category, config)
                    if config.model.save_model:
                        if config.data.category:
                            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                        else:
                            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
                        if not os.path.exists(model_save_dir):
                            os.mkdir(model_save_dir)
                        torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1500))) #config.model.checkpoint_name

                # if epoch %150 == 0  and step ==0: #and epoch>0
                #         validate(model, constants_dict, config, category, v_train)
                #     sample_plot_image(model, trainloader, constants_dict, epoch, category, config)
                    
        if config.model.save_model:
            if config.data.category:
                model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
            else:
                model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, str(config.model.epochs+1500))) #config.model.checkpoint_name
   
        writer.flush()
        writer.close()
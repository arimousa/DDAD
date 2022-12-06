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





def trainer(model, constant_dict, ema_helper, config, category):
    with open('readme.txt', 'a') as f:
        f.write(f"\n {category} : ")
    optimizer = build_optimizer(model, config)
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

    writer = SummaryWriter('runs/DDAD')

    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()


            optimizer.zero_grad()
            loss = get_loss(model, constant_dict, batch[0], t, config) 
            writer.add_scalar('loss', loss, epoch)

            loss.backward()
            optimizer.step()
            if config.model.ema:
                ema_helper.update(model)
            if epoch % 10 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
                with open('readme.txt', 'a') as f:
                    f.write(f"\n Epoch {epoch} | Loss: {loss.item()}  |   ")
            if epoch %50 == 0 and step ==0:
                sample_plot_image(model, trainloader, constant_dict, epoch, category, config)
            if epoch %50 == 0 and epoch > 0 and step ==0:
                validate(model, constant_dict, config, category)
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(config.model.checkpoint_dir, os.path.join(category,str(epoch))), #config.model.checkpoint_name
                )


    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(config.model.checkpoint_dir, os.path.join(category,str(config.model.epochs))), #config.model.checkpoint_name
    )

    writer.flush()
    writer.close()
import torch
import os
import torch.nn as nn
from forward_process import *
from dataset import *

from dataset import *
from test import *
from loss import *
from sample import *


def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    train_dataset = Dataset_maker(
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


    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            optimizer.zero_grad()
            loss = get_loss(model, batch[0], t, config) 
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch %250 == 0 and step ==0:
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)))
                
    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, str(config.model.epochs)))
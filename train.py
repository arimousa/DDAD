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





def trainer(model, constants_dict, v_train, ema_helper, config, category):
    with open('readme.txt', 'a') as f:
        f.write(f"\n {category} : ")
    optimizer = build_optimizer(model, config)
    train_dataset = Dataset(
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


    writer = SummaryWriter('runs/DDAD')

    for epoch in range(config.model.epochs):
        print(f"v_train: {v_train}")
        for step, batch in enumerate(trainloader):
            
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()


            optimizer.zero_grad()
            loss = get_loss(model, constants_dict, batch[0], t, v_train, config) 
#            writer.add_scalar('loss', loss, epoch)

            loss.backward()
            optimizer.step()
            if config.model.ema:
                ema_helper.update(model)
            if epoch % 150 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
                with open('readme.txt', 'a') as f:
                    f.write(f"\n Epoch {epoch} | Loss: {loss.item()}  |   ")
            # if epoch %50 == 0 and step ==0:
            #     sample_plot_image(model, trainloader, constant_dict, epoch, category, config)
            if epoch %150 == 0  and step ==0: #and epoch>0
                for v_test in [70]:
                    print('v_test : ',v_test,'\n')
                    with open('readme.txt', 'a') as f:
                        f.write(f'v_test : {v_test} \n')
                    validate(model, constants_dict, config, category, v_train)
                sample_plot_image(model, trainloader, constants_dict, epoch, category, config)
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(config.model.checkpoint_dir, category,str(f'{epoch}+{v_train}'))) #config.model.checkpoint_name


    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(config.model.checkpoint_dir, category, str(config.model.epochs))) #config.model.checkpoint_name

    writer.flush()
    writer.close()
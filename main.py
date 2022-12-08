import torch
import numpy as np
import os
import argparse
import time
from model import *
from test import validate
from omegaconf import OmegaConf
from utilities import *
import torch.nn.functional as F
from train import trainer
from datetime import timedelta
from EMA import EMAHelper



def constant(config):
    # Define beta schedule

    betas = beta_schedule(beta_schedule = config.model.schedule, beta_start = config.model.beta_start, beta_end=config.model.beta_end, num_diffusion_timesteps=config.model.trajectory_steps)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    constants_dict = {
        'betas' : betas,
        'alphas': alphas,
        'alphas_cumprod' : alphas_cumprod,
        'alphas_cumprod_prev' : alphas_cumprod_prev,
        'sqrt_recip_alphas' : sqrt_recip_alphas,
        'sqrt_alphas_cumprod' : sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod' : sqrt_one_minus_alphas_cumprod,
        'posterior_variance' : posterior_variance,
    }
    return constants_dict


def build_model(config):
    #model = SimpleUnet()
    model = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=3)
    return model


def train(args, category):
    config = OmegaConf.load(args.config)
    
    model = build_model(config)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model.train()
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
    else:
        ema_helper = None
 #   model = torch.nn.DataParallel(model)
    constants_dict = constant(config)
    for v in [0,10,20,30,40,50,60,70,80,90,100]:
        start = time.time()
        print('v_train : ',v,'\n')
        with open('readme.txt', 'a') as f:
            f.write(f'v_train : v \n')
        trainer(model, constants_dict, v, ema_helper, config, category)
        end = time.time()
        print('training time on ',config.model.epochs,' epochs is ', str(timedelta(seconds=end - start)),'\n')
    with open('readme.txt', 'a') as f:
        f.write('\n training time is {}\n'.format(str(timedelta(seconds=end - start))))


def evaluate(args, category):
    start = time.time()
    config = OmegaConf.load(args.config)
    model = build_model(config)
    checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir),os.path.join(category,str(250)))) # config.model.checkpoint_name
    model.load_state_dict(checkpoint)    
    model.to(config.model.device)
    model.eval()
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(checkpoint)
        ema_helper.ema(model)
    else:
        ema_helper = None
    constants_dict = constant(config)
    for v in [0,10,20,30,40,50,60,70,80,90,100]:
        print('v_test : ',v,'\n')
        with open('readme.txt', 'a') as f:
            f.write(f'v_test : {v} \n')
        validate(model, constants_dict, config, category, v)
    end = time.time()
    print('Test time is ', str(timedelta(seconds=end - start)))





def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--eval', 
                                default= False, 
                                help='only evaluate the model')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":

    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    if args.eval:
        print('only evaluation, not training')
        for category in [ 'hazelnut', 'bottle', 'cable', 'carpet',  'leather', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']:
            evaluate(args, category)
    else:
        for category in [ 'hazelnut', 'bottle', 'cable', 'carpet',  'leather', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']:
            print(category)
            train(args, category)
            evaluate(args, category)

        
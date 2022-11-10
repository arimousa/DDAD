import torch
import numpy as np
import os
import argparse
import time
from model import *
from test import validate
from omegaconf import OmegaConf
from utilities import linear_beta_schedule
import torch.nn.functional as F
from train import trainer
from datetime import timedelta



def constant(config):
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=config.model.trajectory_steps)
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
    model = SimpleUnet()
    # model = UNet()
    return model


def train(args, category):
    config = OmegaConf.load(args.config)
    start = time.time()
    model = build_model(config)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print('Beginning mem: ', torch.cuda.memory_allocated(config.model.device))
    model.to(config.model.device)
    print('1- After model to device: ', torch.cuda.memory_allocated(config.model.device))

    model.train()
    constants_dict = constant(config)
    trainer(model, constants_dict, config, category)
    end = time.time()
    print('training time on ',config.model.epochs,' epochs is ', str(timedelta(seconds=end - start)))
    with open('readme.txt', 'w') as f:
        f.write('training time is {}'.format(str(timedelta(seconds=end - start))))
        f.write('\n')


def evaluate(args, category):
    start = time.time()
    config = OmegaConf.load(args.config)
    model = build_model(config)
    checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir),category)) # config.model.checkpoint_name
    model.load_state_dict(checkpoint)    
    model.to(config.model.device)
    model.eval()
    constants_dict = constant(config)
    validate(model, constants_dict, config, category)
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
        for category in ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']:
            evaluate(args, category)
    else:
        for category in ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']:
            train(args, category)
            evaluate(args, category)

        
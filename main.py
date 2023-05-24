import torch
import numpy as np
import os
import argparse
from unet import *
from test import evaluate
from omegaconf import OmegaConf
from train import trainer
from feature_extractor import * 

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"

def build_model(config):
    unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.imput_channel)
    return unet

def train(args):
    config = OmegaConf.load(args.config)
    unet = build_model(config)
    print("Num params: ", sum(p.numel() for p in unet.parameters()))
    unet = unet.to(config.model.device)
    unet.train()
    unet = torch.nn.DataParallel(unet)
    trainer(unet, config.data.category, config)



def test(args):
    config = OmegaConf.load(args.config)
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()
    evaluate(unet, config)



def domain_adaptation(args):
    config = OmegaConf.load(args.config)
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()
    Domain_adaptation(unet, config, fine_tune=True)





def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train the diffusion model')
    cmdline_parser.add_argument('--eval', 
                                default= False, 
                                help='Evaluate the model')
    cmdline_parser.add_argument('--domain_adaptation', 
                                default= False, 
                                help='Domain adaptation')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.train:
        print('Training...')
        train(args)
    if args.domain_adaptation:
        print('Domain Adaptation...')
        domain_adaptation(args)
    if args.eval:
        print('Evaluating...')
        test(args)


        
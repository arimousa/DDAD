import torch
import numpy as np
import os
import argparse
from unet import *
from omegaconf import OmegaConf
from train import trainer
from feature_extractor import * 
from ddad import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def build_model(config):
    if config.model.DDADS:
        unet = UNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2 ,in_channels=config.data.input_channel)
    else:
        unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
    return unet

def train(config):
    torch.manual_seed(42)
    np.random.seed(42)
    unet = build_model(config)
    print(" Num params: ", sum(p.numel() for p in unet.parameters()))
    unet = unet.to(config.model.device)
    unet.train()
    unet = torch.nn.DataParallel(unet)
    # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'1000'))
    # unet.load_state_dict(checkpoint)  
    trainer(unet, config.data.category, config)#config.data.category, 


def detection(config):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    unet.eval()
    ddad = DDAD(unet, config)
    ddad()
    

def finetuning(config):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()
    domain_adaptation(unet, config, fine_tune=True)





def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train the diffusion model')
    cmdline_parser.add_argument('--detection', 
                                default= False, 
                                help='Detection anomalies')
    cmdline_parser.add_argument('--domain_adaptation', 
                                default= False, 
                                help='Domain adaptation')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
    print("Class: ",config.data.category, "   w:", config.model.w, "   v:", config.model.v, "   load_chp:", config.model.load_chp,   "   feature extractor:", config.model.feature_extractor,"         w_DA: ",config.model.w_DA,"         DLlambda: ",config.model.DLlambda)
    print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.train:
        print('Training...')
        train(config)
    if args.domain_adaptation:
        print('Domain Adaptation...')
        finetuning(config)
    if args.detection:
        print('Detecting Anomalies...')
        detection(config)


        
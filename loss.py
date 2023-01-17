import torch
import os
import torch.nn as nn
from forward_process import *
from backbone import *
from noise import *


def get_loss(model, constant_dict, x_0, t, config):

    # x_0 = x_0.to(config.model.device)
    # b = constant_dict['betas'].to(config.model.device)
    # x, e = forward_diffusion_sample(x_0, t, constant_dict, config)
    # # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(config.model.device)
    # # x = x_0 * a.sqrt() + e * (1.0 - a).sqrt()
    # output = model(x, t.float())

    # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


    x_0 = x_0.to(config.model.device)
    b = constant_dict['betas'].to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(config.model.device)
    x = x_0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())

    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
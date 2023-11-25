import torch
import torch.nn as nn
import numpy as np


def get_loss(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


    x = at.sqrt() * x_0 + (1- at).sqrt() * e 
    output = model(x, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


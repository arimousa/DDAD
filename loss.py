import torch
import os
import torch.nn as nn
from forward_process import *
from backbone import *
from noise import *
from visualize import visualalize_distance


def get_loss(model, constant_dict, x_0, t, config):

    x_0 = x_0.to(config.model.device)
    b = constant_dict['betas'].to(config.model.device)
    # x, e = forward_diffusion_sample(x_0, t, constant_dict, config)
    e = torch.randn_like(x_0, device = x_0.device)
    # at = compute_alpha(b, t.long(),config)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(config.model.device)
    # x = x_0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())

    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def get_loss_condition(model, model_y, constant_dict, x_0, y_0, t, config):
    # x_0 = x_0.to(config.model.device)
    # b = constant_dict['betas'].to(config.model.device)
    # e = torch.randn_like(x_0, device = x_0.device)

    # at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # # at_next = (1-b).cumprod(dim=0).index_select(0, t-config.model.skip).view(-1, 1, 1, 1)
    # # c1 =  ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    # xt = at.sqrt() * x_0 + (1- at).sqrt() * e
    # # y = at.sqrt() * x_0 + (1- at_next).sqrt() * e 
    # et = model(xt, t.float())
    # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    # x0_t = x0_t.to(config.model.device)


    # # c1 = (
    # #     config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    # # )
    # # c2 = ((1 - at_next) - c1 ** 2).sqrt()
    
    # # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et

    # visualalize_distance(x0_t.detach() , x_0.detach())

    # drift = model_y(x0_t, x_0 , t) #calculate the drift of the next time step
    # dis = x_0 - x0_t #ground truth
    # return (dis - drift).square().sum(dim=(1, 2, 3)).mean(dim=0)








    x_0 = x_0.to(config.model.device)
    y_0 = y_0.to(config.model.device)
    b = constant_dict['betas'].to(config.model.device)
    # x, e = forward_diffusion_sample(x_0, t, constant_dict, config)
    e = torch.randn_like(x_0, device = x_0.device)
    # e2 = torch.randn_like(y_0, device = y_0.device)
    # print(e.mean(), e2.mean())

    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # at_next = (1-b).cumprod(dim=0).index_select(0, t-config.model.skip).view(-1, 1, 1, 1)
    # c1 =  ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    x = at.sqrt() * x_0 + (1- at).sqrt() * e 
    y = at.sqrt() * y_0 + (1- at).sqrt() * e 

    drift = model_y(x, y , t) #calculate the drift of the next time step
    dis = y - x #ground truth
    return (dis - drift).square().sum(dim=(1, 2, 3)).mean(dim=0)



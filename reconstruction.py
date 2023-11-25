from typing import Any
import torch
# from forward_process import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class Reconstruction:
    '''
    The reconstruction process
    :param y: the target image
    :param x: the input image
    :param seq: the sequence of denoising steps
    :param unet: the UNet model
    :param x0_t: the prediction of x0 at time step t
    '''
    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config

    
    
    def __call__(self, x, y0, w) -> Any:
        def _compute_alpha(t):
            betas = np.linspace(self.config.model.beta_start, self.config.model.beta_end, self.config.model.trajectory_steps, dtype=np.float64)
            betas = torch.tensor(betas).type(torch.float).to(self.config.model.device)
            beta = torch.cat([torch.zeros(1).to(self.config.model.device), betas], dim=0)
            beta = beta.to(self.config.model.device)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            return a
        
        test_trajectoy_steps = torch.Tensor([self.config.model.test_trajectoy_steps]).type(torch.int64).to(self.config.model.device).long()
        at = _compute_alpha(test_trajectoy_steps)
        xt = at.sqrt() * x + (1- at).sqrt() * torch.randn_like(x).to(self.config.model.device)
        seq = range(0 , self.config.model.test_trajectoy_steps, self.config.model.skip)


        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [xt]
            for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(self.config.model.device)
                next_t = (torch.ones(n) * j).to(self.config.model.device)
                at = _compute_alpha(t.long())
                at_next = _compute_alpha(next_t.long())
                xt = xs[-1].to(self.config.model.device)
                self.unet = self.unet.to(self.config.model.device)
                et = self.unet(xt, t)
                yt = at.sqrt() * y0 + (1- at).sqrt() *  et
                et_hat = et - (1 - at).sqrt() * w * (yt-xt)
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
                c1 = (
                    self.config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
                xs.append(xt_next)
        return xs

         




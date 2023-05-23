import torch
from forward_process import *
import numpy as np




def Reconstruction(y0, x, seq, model, config, w):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(t.long(),config)
            at_next = compute_alpha(next_t.long(),config)
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            yt = at.sqrt() * y0 + (1- at).sqrt() *  et
            et_hat = et - (1 - at).sqrt() * w * (yt-xt)
            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next.to('cpu'))
    return xs



def compute_alpha(t, config):
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to(config.model.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


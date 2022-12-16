from asyncio import constants
import torch
from model import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from backbone import *
from metrics import metric

from EMA import EMAHelper
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def validate(model, constants_dict, config, category, v):

    test_dataset = Dataset(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=False,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= config.data.batch_size,
        shuffle=False,
        num_workers= config.model.num_workers,
        drop_last=False,
    )
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
    
    n_saples = 0
    if config.model.representation_backbone == 'cait_m48_448':
        feature_extractor = Feature_extractor(config = config, backbone = "cait_m48_448")
        feature_extractor.to(config.model.device)
        mean_train_dataset = torch.zeros([768, 28, 28]).to(config.model.device)
    else:
        feature_extractor = Feature_extractor(config = config, backbone = "wide_resnet50_2", out_indices=[1])
        feature_extractor.to(config.model.device)
        mean_train_dataset = torch.zeros([256, 64, 64]).to(config.model.device)
    for step, batch in enumerate(trainloader):
        if config.model.representation_backbone == 'cait_m48_448':
            batch[0] = F.interpolate(batch[0] , size = 448, mode="bilinear")
        batch_features = feature_extractor(batch[0].to(config.model.device))
        batch_features = torch.mean(batch_features, dim=0)
        batch_features = batch_features.to(config.model.device)
        mean_train_dataset += batch_features
        n_saples += batch[0].shape[0]
    mean_train_dataset /= n_saples

 
    labels_list = []
    predictions= []
    anomaly_map_list = []
    GT_list = []
    index = 0
    for data, targets, labels in testloader:
        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)
        
        data_forward = []
        data_reconstructed = []

        noisy_image = forward_diffusion_sample(data, test_trajectoy_steps, constants_dict, config)[0]

        seq = range(0, config.model.test_trajectoy_steps, config.model.skip)
        _, data_reconstructed = generalized_steps(noisy_image, seq, model, constants_dict['betas'], config, eta=config.model.eta)
        data_reconstructed = data_reconstructed[-3:-1]
        
        anomaly_map = heat_map(data_reconstructed, data, config, v, mean_train_dataset, representation_backbone = config.model.representation_backbone)

        for pred, label in zip(anomaly_map, labels):
            labels_list.append(0 if label == 'good' else 1)
            predictions.append( torch.max(pred).item())

        if category == None:
            category = 'empty'
        visualize(data, data_reconstructed[-1], targets, anomaly_map, index, category) 
        index = index + data.shape[0] 

        anomaly_map_list.append(anomaly_map)
        GT_list.append(targets)
    
    threshold = metric(labels_list, predictions, anomaly_map_list, GT_list, config)
    print('threshold: ', threshold)


#https://github.com/ermongroup/ddim
def generalized_steps(x, seq, model, b, config, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long(),config)
            at_next = compute_alpha(b, next_t.long(),config)
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def compute_alpha(beta, t, config):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    beta = beta.to(config.model.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



#https://github.com/NVlabs/edm/blob/main/generate.py
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
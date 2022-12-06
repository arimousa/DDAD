import torch
from torch.optim import Adam

def build_optimizer(model, config):
    lr = config.model.learning_rate
    weight_decay = config.model.weight_decay
    return Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
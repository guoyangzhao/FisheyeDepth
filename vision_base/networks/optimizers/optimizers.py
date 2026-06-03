import torch.nn as nn
import torch.optim as optim

def build_optimizer(model:nn.Module, name, **kwargs):
    if name.lower() == 'sgd':
        return optim.SGD(model.parameters(), **kwargs)
    if name.lower() == 'adam':
        return optim.Adam(model.parameters(), **kwargs)
    if name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), **kwargs)
    raise NotImplementedError(name)

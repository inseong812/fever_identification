import torch.optim as optim

def get_optimizer(optimizer_name, parameters, **kwargs):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(parameters, **kwargs)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(parameters, **kwargs)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(parameters, **kwargs)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(parameters, **kwargs)
    return optimizer

import torch

def save_models(output_path, model, optimizer):
    is_distributed = torch.distributed.is_initialized()
    torch.save({
            'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, output_path)

def load_models(path, model, optimizer=None, map_location="cuda:0", strict=False):
    checkpoint = torch.load(path, map_location=map_location)

    is_distributed = torch.distributed.is_initialized()
    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

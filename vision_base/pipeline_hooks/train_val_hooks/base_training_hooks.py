from typing import Optional, Dict, List
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer

from vision_base.utils.logger import LossLogger
from vision_base.utils.timer import profile
class BaseTrainingHook(object):
    """
        Base Training hook functions do not have input nor output. It is responsible of running training pipelines
        But they can have initialization parameters.
    """
    def __init__(self,
                 tensor_keys:Optional[List[str]]=None,
                 clip_gradients:Optional[float]=None,
                 **kwargs):
        self.tensor_keys = tensor_keys
        self.clip_gradients = clip_gradients

    @profile('Training hook', 0, 100)
    def __call__(self, data:Dict,
                       meta_arch:nn.Module,
                       optimizer:Optimizer,
                       writer:Optional[SummaryWriter]=None,
                       training_loss_logger:Optional[LossLogger]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        optimizer.zero_grad()

        for key in data:
            if isinstance(data[key], torch.Tensor):
                if self.tensor_keys is None or key in self.tensor_keys:
                    data[key] = data[key].cuda().contiguous()

        meta = dict(epoch_num=epoch_num, global_step=global_step, is_training=True)
        output:dict = meta_arch(data, meta)

        if training_loss_logger is not None:
            training_loss_logger.update(output['loss_dict'])
            training_loss_logger.update_hm(output.get('hm', dict()))

        output['loss'].mean().backward()

        if self.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(meta_arch.parameters(), self.clip_gradients)
        
        optimizer.step()

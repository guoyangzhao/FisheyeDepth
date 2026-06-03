from typing import Optional, Dict, List
import torch
import torch.nn as nn

class BaseValidationHook(object):
    """
        Base Validation hook functions do not have input but with dictionary output. It is responsible of running testing pipelines for evaluation functions.
        But they can have initialization parameters.
    """
    def __init__(self,
                 tensor_keys:Optional[List[str]]=None,
                 **kwargs):
        self.tensor_keys = tensor_keys

    def __call__(self, data:Dict,
                       meta_arch:nn.Module,
                       global_step:int=0,
                       epoch_num:int=0
                       ) -> Dict:
        for key in data:
            if isinstance(data[key], torch.Tensor):
                if self.tensor_keys is None or key in self.tensor_keys:
                    data[key] = data[key].cuda().contiguous()

        meta = dict(epoch_num=epoch_num, global_step=global_step, is_training=False)
        output = meta_arch(data, meta)

        return output

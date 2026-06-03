from tqdm import tqdm
from easydict import EasyDict
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset # noqa: F401
from torch.utils.tensorboard import SummaryWriter
from vision_base.utils.builder import build
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.pipeline_hooks.train_val_hooks.base_validation_hooks import BaseValidationHook

class BaseEvaluationHook(object):
    """
        Base Evaluation hook functions do not have input nor output. It is responsible of running evaluation pipelines
        But they can have initialization parameters.
    """
    def __init__(self,
                test_run_hook_cfg:EasyDict,
                dataset_eval_cfg:EasyDict, #dataset specific
                result_path_split:str='validation', # determine in train/test code, should not in config
                **kwargs):
        self.test_hook:BaseValidationHook = build(**test_run_hook_cfg)
        self.result_path_split = result_path_split
        self.dataset_eval = build(**dataset_eval_cfg)
        for key in kwargs:
            setattr(self, key, kwargs[key])


    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        self.dataset_eval.reset()
        
        for index in tqdm(range(len(dataset_val)), dynamic_ncols=True):
            data = dataset_val[index]
            collated_data:Dict = collate_fn([data])

            output_dict = self.test_hook(collated_data, meta_arch, global_step, epoch_num)

            self.dataset_eval.step(index, output_dict, data)

        if not self.result_path_split == 'test' and self.dataset_eval is not None:
            self.dataset_eval(writer, global_step, epoch_num)

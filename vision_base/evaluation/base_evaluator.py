from typing import Optional
from torch.utils.tensorboard import SummaryWriter

class BaseEvaluator(object):
    def __init__(self,
                 data_path, #path to kitti raw data
                 split_file,
                 gt_saved_file,
                 is_evaluate_absolute=False,
                 ):
        pass

    def reset(self):
        pass

    def step(index, output_dict, data):
        pass

    def log(self, writer, mean_errors, mean_abs_errors, global_step=0, epoch_num=0, is_print=True):
        pass

    def __call__(self, result_path, writer:Optional[SummaryWriter]=None, global_step=0, epoch_num=0):
        pass

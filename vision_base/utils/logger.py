from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LogImageStruct(object):
    def __init__(self, data:torch.Tensor, dataformat:str='NCHW'):
        self.data = data
        self.dataformat = dataformat

    def update(self, data:torch.Tensor):
        self.data = data  # directly cover

    def log_images(self, writer:SummaryWriter, tag:str, *args, **kwargs):
        writer.add_images(tag, self.data, dataformats=self.dataformat, **kwargs)


class LossLogger():
    def __init__(self, recorder, data_split='train'):
        self.recorder = recorder
        self.data_split = data_split
        self.reset()

    def reset(self):
        self.loss_stats:Dict[str, float] = {}  # each will be
        self.hm_stats:Dict[str, LogImageStruct] = {}

    def update(self, loss_dict):
        for key in loss_dict:
            if key not in self.loss_stats:
                self.loss_stats[key] = AverageMeter()
            self.loss_stats[key].update(loss_dict[key].mean().item())

    def log(self, step):
        for key in self.loss_stats:
            name = key + '/' + self.data_split
            self.recorder.add_scalar(name, self.loss_stats[key].avg, step)

        for key in self.hm_stats:
            name = key + '/' + self.data_split
            self.hm_stats[key].log_images(self.recorder, name, global_step=step)

    def update_hm(self, feature_map_dict):
        for key in feature_map_dict:
            if isinstance(feature_map_dict[key], dict):
                data = feature_map_dict[key]['data']
                dataformat = feature_map_dict[key].get('dataformat', 'NCHW')
            else:
                data = feature_map_dict[key]
                if len(data.shape) == 4:
                    dataformat = 'NCHW'
                if len(data.shape) == 3:
                    dataformat = 'CHW'
            if key not in self.hm_stats:
                self.hm_stats[key] = LogImageStruct(data, dataformat)
            else:
                self.hm_stats[key].update(data)


GIT_FORMATTER = \
"""
-----------------
# Git Last Commit
{git_log}

-----------------
# Git Diff

"""


def styling_git_info(repo):
    git_diff = f"```diff\n{repo.git.diff()}\n```"
    git_log = GIT_FORMATTER.format(git_log=repo.git.log(-1)).replace(' ', '&nbsp;').replace('\n', '  \n')
    return f"{git_log}{git_diff}"

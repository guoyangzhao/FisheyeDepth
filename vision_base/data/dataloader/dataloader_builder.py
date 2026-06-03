from typing import Callable
from torch.utils.data import DataLoader
from vision_base.utils.builder import build

def build_dataloader(dataset,
                    num_workers: int,
                    batch_size: int,
                    collate_fn: Callable,
                    local_rank: int = -1,
                    world_size: int = 1,
                    sampler_cfg: dict = dict(),
                    **kwargs):
    sampler_name = sampler_cfg.pop('name') if 'name' in sampler_cfg else 'vision_base.data.dataloader.distributed_sampler.TrainingSampler'
    sampler = build(sampler_name, size=len(dataset), rank=local_rank, world_size=world_size, **sampler_cfg)

    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler, **kwargs, drop_last=True)
    return dataloader

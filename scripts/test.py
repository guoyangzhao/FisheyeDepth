
import fire
import torch

from _path_init import manage_package_logging
from vision_base.utils.builder import build
from vision_base.utils.utils import cfg_from_file, update_cfg
from vision_base.networks.utils.utils import load_models

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(config:str="config/config.py",
        gpu:int=0,
        checkpoint_path:str="monodepth.networks.models.meta_archs.monodepth2_model.MonoDepthWPose_latest.pth",    ##gyzhao## monodepth.networks.models.meta_archs.monodepth2_model.MonoDepthWPose_latest.pth   retinanet_79
        split_to_test:str='validation',
        **kwargs):
    # Read Config
    cfg = cfg_from_file(config)
    cfg = update_cfg(cfg, **kwargs)

    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    manage_package_logging()
    
    # Set up dataset and dataloader
    if split_to_test == 'training':
        dataset = build(**cfg.train_dataset)
    elif split_to_test == 'test':
        dataset = build(**cfg.test_dataset)
    else:
        dataset = build(**cfg.val_dataset)

    # Create the model
    meta_arch = build(**cfg.meta_arch)
    meta_arch = meta_arch.cuda()

    load_models(checkpoint_path, meta_arch, map_location=f'cuda:{gpu}', strict=False)
    meta_arch.eval()

    if 'evaluate_hook' in cfg.trainer:
        evaluate_hook = build(result_path_split=split_to_test, **cfg.trainer.evaluate_hook)
        print("Found evaluate function")
    else:
        raise KeyError("evaluate_hook not found in Config")

    # Run evaluation
    evaluate_hook(meta_arch, dataset)
    print('finish')


if __name__ == '__main__':
    fire.Fire(main)

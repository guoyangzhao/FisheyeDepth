"""
    Script for launching training process
"""
import os
import shutil
from easydict import EasyDict
from fire import Fire
import torch
from torch.utils.tensorboard import SummaryWriter

from _path_init import manage_package_logging
from vision_base.utils.builder import build
from vision_base.utils.utils import get_num_parameters, cfg_from_file, set_random_seed, update_cfg
from vision_base.utils.timer import Timer
from vision_base.utils.logger import LossLogger, styling_git_info
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.data.dataloader import build_dataloader
from vision_base.networks.optimizers import optimizers, schedulers
from vision_base.networks.utils.utils import save_models, load_models

def main(config="configs/config.py", experiment_name="default", world_size=1, **kwargs):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
        experiment_name (str): Custom name for the experitment, only used in tensorboard.
        world_size (int): Number of total subprocesses in distributed training.
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training.
    """

    ## Get config
    cfg = cfg_from_file(config)
    cfg = update_cfg(cfg, **kwargs)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()
    cfg.dist.world_size = world_size
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    ## Setup writer if local_rank > 0
    recorder_dir = os.path.join(cfg.path.log_path, experiment_name + f"config={config}")
    if is_logging: # writer exists only if not distributed and local rank is smaller
        ## Clean up the dir if it exists before
        if os.path.isdir(recorder_dir):
            shutil.rmtree(recorder_dir, ignore_errors=True)
            print("clean up the recorder directory of {}".format(recorder_dir))
        writer = SummaryWriter(recorder_dir)

        ## Record config object using pprint
        import pprint

        formatted_cfg = pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text

    #     ## Record Git status
    #     import git
    #     repo = git.Repo(cfg.path.base_path)
    #     writer.add_text("git/git_show", styling_git_info(repo))
    #     writer.flush()
    # else:
    #     writer = None

    ## Set up GPU and distribution process
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    set_random_seed(123)
    torch.cuda.set_device(gpu)
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(local_rank)

    ## Precomputing Hooks
    if 'precompute_hook' in cfg:
        precompute_hook = build(**cfg.precompute_hook)
        precompute_hook()
 
    ## define datasets and dataloader.
    dataset_train = build(**cfg.train_dataset)
    dataset_val = build(**cfg.val_dataset)

    dataloader_train = build_dataloader(dataset_train,
                                        num_workers=cfg.data.num_workers,
                                        batch_size=cfg.data.batch_size,
                                        collate_fn=collate_fn,
                                        local_rank=local_rank,
                                        world_size=world_size,
                                        sampler_cfg=getattr(cfg.data, 'sampler', dict()))

    ## Create the model
    meta_arch = build(**cfg.meta_arch)
    from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
    assert isinstance(meta_arch, BaseMetaArch)

    ## Convert to cuda
    if is_distributed:
        meta_arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(meta_arch)
        meta_arch = torch.nn.parallel.DistributedDataParallel(meta_arch.cuda(), device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    else:
        meta_arch = meta_arch.cuda()
    meta_arch.train()

    ## Record basic information of the model
    if is_logging:
        string1 = meta_arch.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("model structure", string1) # add space for markdown style in tensorboard text
        num_parameters = get_num_parameters(meta_arch)
        print(f'number of trained parameters of the model: {num_parameters}')
    
    ## define optimizer and weight decay
    optimizer = optimizers.build_optimizer(meta_arch, **cfg.optimizer)

    ## define scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.trainer.max_epochs, cfg.optimizer.lr_target)
    scheduler_config = getattr(cfg, 'scheduler', None)
    scheduler = schedulers.build_scheduler(optimizer, **scheduler_config)
    is_iter_based = getattr(scheduler_config, "is_iter_based", False)

    ## define loss logger
    training_loss_logger = LossLogger(writer, 'train') if is_logging else None
    manage_package_logging()

    ## Load old model if needed
    old_checkpoint = getattr(cfg.path, 'pretrained_checkpoint', None)
    if old_checkpoint is not None:
        load_models(old_checkpoint,
                    meta_arch,
                    optimizer,
                    map_location=f'cuda:{gpu}')

    ## training pipeline
    if 'training_hook' in cfg.trainer:
        training_hook = build(**cfg.trainer.training_hook)
        from vision_base.pipeline_hooks.train_val_hooks.base_training_hooks import BaseTrainingHook
        assert isinstance(training_hook, BaseTrainingHook)
    else:
        raise KeyError

    ## Get evaluation pipeline
    if is_evaluating and 'evaluate_hook' in cfg.trainer:
        evaluate_hook = build(result_path_split='validation', **cfg.trainer.evaluate_hook)
        from vision_base.pipeline_hooks.evaluation_hooks.base_evaluation_hooks import BaseEvaluationHook
        assert isinstance(evaluate_hook, BaseEvaluationHook)
        print("Found evaluate function {}".format(cfg.trainer.evaluate_hook.name))
    else:
        evaluate_hook = None
        print(f"Evaluate function not found or not used in local_rank={local_rank}")


    ## timer is used to estimate eta
    timer = Timer()

    print('Num training images: {}'.format(len(dataset_train)))

    global_step = 0

    for epoch_num in range(cfg.trainer.max_epochs):
        ## Start training for one epoch
        meta_arch.train()
        if training_loss_logger:
            training_loss_logger.reset()
        for iter_num, data in enumerate(dataloader_train):
            training_hook(data, meta_arch, optimizer, writer, training_loss_logger, global_step, epoch_num)

            global_step += 1

            if is_iter_based:
                scheduler.step()

            if is_logging and global_step % cfg.trainer.disp_iter == 0:
                ## Log loss, print out and write to tensorboard in main process
                if 'total_loss' not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {epoch_num}, iteration:{iter_num}, global_step:{global_step}, total_loss not found in logger.")
                else:
                    log_str = 'Epoch: {} | Iteration: {}  | Running loss: {:1.5f} | eta:{}'.format(
                        epoch_num, iter_num, training_loss_logger.loss_stats['total_loss'].avg,
                        timer.compute_eta(global_step, len(dataloader_train) * cfg.trainer.max_epochs / world_size))
                    print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, global_step)
                    training_loss_logger.log(global_step)

        if not is_iter_based:
            scheduler.step()

        ## save model in main process if needed
        if is_logging:
            save_models(os.path.join(cfg.path.checkpoint_path, f"{cfg.meta_arch.name}_latest.pth"),
                        meta_arch, optimizer)

        if is_logging and (epoch_num + 1) % cfg.trainer.save_iter == 0:
            save_models(os.path.join(cfg.path.checkpoint_path, f"{cfg.meta_arch.name}_{epoch_num}.pth"),
                        meta_arch, optimizer)

        ## test model in main process if needed
        if is_evaluating and evaluate_hook is not None and cfg.trainer.test_iter > 0 and (epoch_num + 1) % cfg.trainer.test_iter == 0:
            print("\n/**** start testing after training epoch {} ******/".format(epoch_num))
            evaluate_hook(meta_arch.module if is_distributed else meta_arch, dataset_val, writer, epoch_num, epoch_num)
            print("/**** finish testing after training epoch {} ******/".format(epoch_num))

        if is_distributed:
            # torch.distributed.barrier() # wait untill all finish a epoch
            if isinstance(dataloader_train.sampler, torch.utils.data.DistributedSampler):
                dataloader_train.sampler.set_epoch(epoch_num)

        if is_logging:
            writer.flush()


if __name__ == '__main__':
    Fire(main)

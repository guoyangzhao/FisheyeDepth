from easydict import EasyDict as edict
import os
import numpy as np

cfg = edict()



## path
def build_path():
    path = edict()
    path.kitti360_path = "/home/gyzhao/4T/FisheyeDepth/FisheyeDepth_dataset/KITTI-360"   ### Dataset path
    path.resized_root = "/home/gyzhao/4T/FisheyeDepth/data_shared/Docker-yxliu/kitti360_resized_fisheye"  ### path for preprocessed resized images
    path.base_path = "/home/gyzhao/fisheyedepth"   ### Base path for project
    path.project_path = "/home/gyzhao/fisheyedepth/workdirs"   ### Base path for saving logs, checkpoints and preprocessed data
    
    if not os.path.isdir(path.project_path):
        os.mkdir(path.project_path)
    path.project_path = os.path.join(path.project_path, 'fisheye-test')
    if not os.path.isdir(path.project_path):
        os.mkdir(path.project_path)

    path.log_path = os.path.join(path.project_path, "log")
    if not os.path.isdir(path.log_path):
        os.mkdir(path.log_path)

    path.checkpoint_path = os.path.join(path.project_path, "checkpoint")
    if not os.path.isdir(path.checkpoint_path):
        os.mkdir(path.checkpoint_path)

    path.preprocessed_path = os.path.join(path.project_path, "output")
    if not os.path.isdir(path.preprocessed_path):
        os.mkdir(path.preprocessed_path)

    path.train_imdb_path = os.path.join(path.preprocessed_path, "training")
    if not os.path.isdir(path.train_imdb_path):
        os.mkdir(path.train_imdb_path)

    path.val_imdb_path = os.path.join(path.preprocessed_path, "validation")
    if not os.path.isdir(path.val_imdb_path):
        os.mkdir(path.val_imdb_path)
    return path

cfg.path = build_path()

## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 20,
    disp_iter = 50,
    save_iter = 5,
    test_iter = 1,
    training_hook = edict(
        name='vision_base.pipeline_hooks.train_val_hooks.base_training_hooks.BaseTrainingHook',
        clip_gradients=1.0,
    ),
    evaluate_hook = edict(
        name="monodepth.pipeline_hooks.evaluation_hooks.base_evaluation_hooks.KittiEvaluationHook",
        test_run_hook_cfg=edict(name='vision_base.pipeline_hooks.train_val_hooks.base_validation_hooks.BaseValidationHook'),
        dataset_eval_cfg=edict(
            name='monodepth.evaluation.kitti360_fisheye_eval.Kitti360FisheyeEvaluator',
            data_path=cfg.path.kitti360_path,
            split_file=os.path.join(cfg.path.base_path, 'meta_data', 'kitti360_trainsub', 'kitti360_val.txt'),
            gt_saved_file=os.path.join(cfg.path.base_path, 'meta_data', 'kitti360_trainsub', 'fisheye_gt_depth.npz'),
        ),
    )
)
cfg.trainer = trainer

## optimizer
optimizer = edict(
    name = 'adam',
    lr        = 1e-4,
    weight_decay = 1e-5,
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    name = 'StepLR',
    step_size = 8
)
cfg.scheduler = scheduler

data = edict(
    batch_size = 16,
    num_workers = 4,
    rgb_shape = (384, 384, 3),   ### input image RGB shape after augmentation
    frame_idxs  = [0, 1, -1],
)

### use fisheye_dataset for training
train_dataset = edict(
    name = "vision_base.data.datasets.dataset_utils.ConcatDataset",
    frame_idxs = data.frame_idxs,
    is_motion_mask = False,
    is_precompute_flow = False,
    is_filter_static = True,
    resized_root = cfg.path.resized_root,
    cfg_list = [
        edict(
           name = "monodepth.data.datasets.fisheye_dataset.KITTI360FisheyeDataset",
           raw_path = cfg.path.kitti360_path,
           fish_eye_mask = os.path.join(cfg.path.base_path, 'meta_data', 'kitti360_trainsub', 'fisheye_mask.png'),
           split_file = os.path.join(cfg.path.base_path, 'meta_data', 'kitti360_trainsub', 'kitti360_train.txt'),
        )
    ],
)

### use fisheye_dataset for evaluation
val_dataset = edict(
   name = "monodepth.data.datasets.fisheye_dataset.KITTI360FisheyeDataset",  
   raw_path = cfg.path.kitti360_path,
   ### validation split
   split_file = os.path.join(cfg.path.base_path, 'meta_data', 'kitti360_trainsub', 'kitti360_val.txt'),    ## Validation split
   is_filter_static = False,
   use_right_image=False,   ## KITTI360 dataset, choose left-cam or right-cam
   frame_ids=[0],
)

### Build Augmentation
resize_image_keys=[('image', idx) for idx in data.frame_idxs]
original_image_keys = [('original_image', idx) for idx in data.frame_idxs]
pose_axis_pairs = [
    (("relative_pose", idx), 0) for idx in data.frame_idxs[1:]
]
data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    #crop_top = 100,
    key_mappings=edict(
        image_keys=resize_image_keys,
        calib_keys=['P2'],
        gt_image_keys=['patched_mask'],
    )
)
aug_path='vision_base.data.augmentations.augmentations'
train_dataset.augmentation = edict(
    name='vision_base.utils.builder.Sequential',
    cfg_list = [
        edict(name=f'{aug_path}.ConvertToFloat'),
        edict(name=f'{aug_path}.Resize', size=data.augmentation.cropSize, preserve_aspect_ratio=True, force_pad=True),
        edict(name=f'{aug_path}.RandomMirror', mirror_prob=0.5, pose_axis_pairs=pose_axis_pairs),
        edict(name=f'{aug_path}.Copy', from_keys=resize_image_keys, to_keys=original_image_keys),
        edict(name="vision_base.utils.builder.Shuffle", 
              cfg_list=[
                    edict(name=f"{aug_path}.RandomBrightness", distort_prob=1.0),
                    edict(name=f"{aug_path}.RandomContrast", distort_prob=1.0, lower=0.6, upper=1.4),
                    edict(name="vision_base.utils.builder.Sequential",
                        cfg_list=[
                            edict(name=f"{aug_path}.ConvertColor", transform='HSV'),
                            edict(name=f"{aug_path}.RandomSaturation", distort_prob=1.0, lower=0.6, upper=1.4),
                            edict(name=f"{aug_path}.ConvertColor", current='HSV', transform='RGB'),
                        ] 
                    )
            ],
        ),
        edict(name=f'{aug_path}.Normalize', mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std, image_keys=resize_image_keys),
        edict(name=f'{aug_path}.Normalize', mean=np.array([0, 0, 0]), stds=np.array([1, 1, 1]), image_keys=[('original_image', idx) for idx in data.frame_idxs]),
        edict(name=f'{aug_path}.ConvertToTensor', image_keys=resize_image_keys+original_image_keys),
        
    ],
    **data.augmentation.key_mappings # common keywords
)

val_dataset.augmentation = edict(
    name='vision_base.utils.builder.Sequential',
    cfg_list=[
            edict(name=f'{aug_path}.ConvertToFloat'),
            edict(name=f'{aug_path}.Resize', size=data.augmentation.cropSize, preserve_aspect_ratio=False), #preserve_aspect_ratio=True, force_pad=True
            edict(name=f'{aug_path}.Normalize', mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std),
            edict(name=f'{aug_path}.ConvertToTensor'),
    ],
    image_keys=[('image', 0)], 
    calib_keys=['P2']
)

cfg.data = data
cfg.train_dataset = train_dataset
cfg.val_dataset = val_dataset

## networks
meta_arch = edict(
    name='monodepth.networks.models.meta_archs.monodepth2_model.MonoDepthWPose',

    depth_backbone_cfg = edict(
        name='vision_base.networks.models.backbone.resnet.resnet',
        depth=18,
        pretrained=True,
        frozen_stages=-1,
        num_stages=4,
        out_indices=(-1, 0, 1, 2, 3),
        norm_eval=False,
        dilations=(1, 1, 1, 1),
    ),

    head_cfg = edict(
        name='monodepth.networks.models.heads.monodepth2_decoder.FishEyeDecoder',
        scales=[0, 1, 2, 3],
        height=data.rgb_shape[0],
        width=data.rgb_shape[1],
        min_depth=0.5,
        max_depth=150.0,
        overlapped_mask=True,
        depth_decoder_cfg=edict(
            name='monodepth.networks.models.heads.depth_encoder.MultiChannelDepthDecoder', #MultiChannelScaleDepthDecoder',
            num_ch_enc=np.array([64, 64, 128, 256, 512]),
            num_output_channels=64,
            use_skips=True,
            scales=[0, 1, 2, 3],
            min_depth=0.5,
            max_depth=150
        ),
    ),

    train_cfg = edict(
        frame_ids=[0, 1, -1],
    ),

    test_cfg = edict(

    ),
)



cfg.meta_arch = meta_arch
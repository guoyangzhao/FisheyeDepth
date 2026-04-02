import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset # noqa: F401
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import cv2
from monodepth.networks.utils.postopt_utils import read_sparse_vo, post_optimization, depth_image_to_point_cloud_array, denorm
from vision_base.utils.builder import build
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.pipeline_hooks.train_val_hooks.base_validation_hooks import BaseValidationHook
from vision_base.pipeline_hooks.evaluation_hooks.base_evaluation_hooks import BaseEvaluationHook



class KittiEvaluationHook(BaseEvaluationHook):
    """
        Base Evaluation hook functions do not have input but no output. It is responsible of running evaluation pipelines
        But they can have initialization parameters.
    """
    def __init__(self,
                test_run_hook_cfg:EasyDict,
                dataset_eval_cfg:EasyDict, #dataset specific
                **kwargs):
        self.test_hook:BaseValidationHook = build(**test_run_hook_cfg)
        self.dataset_eval_func = build(**dataset_eval_cfg)
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

        batch_size = getattr(self, 'batch_size', 1)
        num_workers = getattr(self, 'num_workers', 4)
        dataloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

        errors:List[float]= []
        abs_errors:List[float] = []
        frame_index = 0
        for batched_data in tqdm(dataloader):
            output_dict = self.test_hook(batched_data, meta_arch, global_step, epoch_num)
            B = output_dict['depth'].shape[0]
            for i in range(B):
                depth =output_dict["depth"][i, 0]
                h_eff, w_eff = batched_data[('image_resize', 'effective_size')][i]
                depth = depth[0:h_eff, 0:w_eff]
                h, w = batched_data[('image_resize', 'original_shape')][i]
                h = h.item()
                w = w.item()
                depth_0 = 1 / cv2.resize(1 / depth.cpu().numpy(), (w, h))

                return_dict = self.dataset_eval_func.single_call(depth_0, frame_index)
                frame_index += 1
                errors.append(return_dict['error'])
                abs_errors.append(return_dict['abs_error'])

        mean_errors = np.array(errors).mean(0)
        mean_abs_errors = np.array(abs_errors).mean(0)
        self.dataset_eval_func.log(writer, mean_errors, mean_abs_errors, global_step=global_step, epoch_num=epoch_num)

class KittiEvaluationHook_postopt(KittiEvaluationHook):
    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        post_opt_cfg = getattr(self, 'post_opt_cfg', dict())
        print(post_opt_cfg)
        vo_path = getattr(post_opt_cfg, 'vo_path', None)
        post_opt_param_dict = dict(
            lab_dist_weight=1,
            depth_dist_weight=1,
            image_dist_weight=1,
            h_seg=10,
            w_seg=18,
            iter_num=3,
            lambda0 = 0.54/(10*18),
            lambda1 = 1.0,
            lambda2 = 0.4
        )
        for key in post_opt_param_dict:
            if key in post_opt_cfg:
                post_opt_param_dict[key] = post_opt_cfg[key]

        batch_size = getattr(self, 'batch_size', 1)
        num_workers = getattr(self, 'num_workers', 4)
        dataloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

        errors = []
        abs_errors = []
        frame_index = 0
        for batched_data in tqdm(dataloader):
            image = batched_data[('image', 0)]
            output_dict = self.test_hook(batched_data, meta_arch, global_step, epoch_num)
            B = output_dict['depth'].shape[0]
            for i in range(B):
                depth =output_dict["depth"][i, 0]
                h_eff, w_eff = batched_data[('image_resize', 'effective_size')][i]
                depth = depth[0:h_eff, 0:w_eff]

                rgb_image = denorm(image[0].cpu().numpy().transpose([1, 2, 0]),
                                 rgb_mean=np.array([0.485, 0.456, 0.406]),
                                 rgb_std=np.array([0.229, 0.224, 0.225]),)
        
                if ('vo_depth', 0) in batched_data:
                    sub_depth_map = batched_data[('vo_depth', 0)][0]
                else:
                    try:
                        sub_depth_map = read_sparse_vo(dataset_val, frame_index,  rgb_image.shape[0], rgb_image.shape[1], vo_folder=vo_path)
                        projected_depth_map = depth_image_to_point_cloud_array(depth.cpu().numpy())
                        depth = post_optimization(rgb_image, projected_depth_map, depth, sub_depth_map,
                                **post_opt_param_dict)
                    except: # noqa: 722
                        depth = depth

                
                h, w, _ = batched_data[('original_image', 0)][i].shape
                depth_0 = 1 / cv2.resize(1 / depth.cpu().numpy(), (w, h))

                return_dict = self.dataset_eval_func.single_call(depth_0, frame_index)
                frame_index += 1
                errors.append(return_dict['error'])
                abs_errors.append(return_dict['abs_error'])

        mean_errors = np.array(errors).mean(0)
        mean_abs_errors = np.array(abs_errors).mean(0)
        self.dataset_eval_func.log(writer, mean_errors, mean_abs_errors, global_step=global_step, epoch_num=epoch_num)

class FastNuscEvaluationHook(BaseEvaluationHook):
    def __init__(self,
                test_run_hook_cfg:EasyDict,
                dataset_eval_cfg:EasyDict, #dataset specific
                **kwargs):
        self.test_hook:BaseValidationHook = build(**test_run_hook_cfg)
        self.dataset_eval_func = build(**dataset_eval_cfg)
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

        batch_size = getattr(self, 'batch_size', 16)
        num_workers = getattr(self, 'num_workers', 4)
        dataloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

        errors:Dict[str, List] = dict()
        abs_errors:Dict[str, List] = dict()
        for batched_data in tqdm(dataloader):
            output_dict = self.test_hook(batched_data, meta_arch, global_step, epoch_num)
            B = output_dict['depth'].shape[0]
            for i in range(B):
                depth = output_dict["depth"][i, 0]
                h_eff, w_eff = batched_data[('image_resize', 'effective_size')][i]
                depth = depth[0:h_eff, 0:w_eff]
                h, w, _ = batched_data[('original_image', 0)][i].shape
                depth_0 = cv2.resize(depth.cpu().numpy(), (w, h))

                camera_type = batched_data['camera_type'][i]
                if camera_type not in errors:
                    errors[camera_type] = []
                    abs_errors[camera_type] = []
                if self.dataset_eval_func is not None:
                    filename = batched_data[('filename', 0)][i]
                    try:
                        return_dict = self.dataset_eval_func.single_call(depth_0, filename)
                    except ValueError:
                        import warnings
                        warnings.warn(f"image at sample {filename}  has no usable points")
                        continue
                    errors[camera_type].append(return_dict['error'])
                    abs_errors[camera_type].append(return_dict['abs_error'])

        all_mean_errors = []
        all_mean_errors_abs = []
        for cam in errors:
            mean_errors = np.array(errors[cam]).mean(0)
            mean_abs_errors = np.array(abs_errors[cam]).mean(0)
            if self.dataset_eval_func is not None:
                self.dataset_eval_func.log(writer, cam, mean_errors, mean_abs_errors, global_step=global_step, epoch_num=epoch_num)
            all_mean_errors.append(mean_errors)
            all_mean_errors_abs.append(mean_abs_errors)
        all_mean_errors = np.array(all_mean_errors).mean(0)
        all_mean_errors_abs = np.array(all_mean_errors_abs).mean(0)
        if self.dataset_eval_func is not None:
            self.dataset_eval_func.log(writer, 'all mean', all_mean_errors, all_mean_errors_abs, global_step=global_step, epoch_num=epoch_num)

class PostOptFastNuscEvaluationHook(FastNuscEvaluationHook):
    def _init_post_opt(self):
        post_opt_cfg = getattr(self, 'post_opt_cfg', dict())
        post_opt_param_dict = dict(
            lab_dist_weight=1,
            depth_dist_weight=1,
            image_dist_weight=1,
            h_seg=10,
            w_seg=18,
            iter_num=3,
            lambda0 = 0.54/(10*18),
            lambda1 = 1.0,
            lambda2 = 0.4
        )
        for key in post_opt_param_dict:
            if key in post_opt_cfg:
                post_opt_param_dict[key] = post_opt_cfg[key]
        return post_opt_param_dict

    @torch.no_grad()
    def __call__(self, meta_arch:nn.Module,
                       dataset_val,
                       writer:Optional[SummaryWriter]=None,
                       global_step:int=0,
                       epoch_num:int=0
                       ):
        meta_arch.eval()
        post_opt_param_dict = self._init_post_opt()
        batch_size = getattr(self, 'batch_size', 16)
        num_workers = getattr(self, 'num_workers', 4)
        dataloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

        errors:Dict[str, List] = dict()
        abs_errors:Dict[str, List] = dict()
        for batched_data in tqdm(dataloader):
            output_dict = self.test_hook(batched_data, meta_arch, global_step, epoch_num)
            B = output_dict['depth'].shape[0]
            for i in range(B):
                depth = output_dict["depth"][i, 0]
                h_eff, w_eff = batched_data[('image_resize', 'effective_size')][i]
                depth = depth[0:h_eff, 0:w_eff]
                h, w, _ = batched_data[('original_image', 0)][i].shape
                

                projected_depth_map = depth_image_to_point_cloud_array(depth.cpu().numpy())
                image = batched_data[('image', 0)]
                rgb_image = denorm(image[i].cpu().numpy().transpose([1, 2, 0]),
                                    rgb_mean=np.array([0.485, 0.456, 0.406]),
                                    rgb_std=np.array([0.229, 0.224, 0.225]),)
                #_, sub_depth_map = read_sparse_depth(dataset_val, index, original_image.shape[0], original_image.shape[1],
                #                                  rgb_image.shape[0], rgb_image.shape[1], subsample_ratio=0.2)
                sub_depth_map = batched_data[('vo_depth', 0)][i]

                depth_0 = post_optimization(rgb_image, projected_depth_map, depth, sub_depth_map,
                                    **post_opt_param_dict)

                depth_0 = cv2.resize(depth_0.cpu().numpy(), (w, h))

                camera_type = batched_data['camera_type'][i]
                if camera_type not in errors:
                    errors[camera_type] = []
                    abs_errors[camera_type] = []
                if self.dataset_eval_func is not None:
                    filename = batched_data[('filename', 0)][i]
                    try:
                        return_dict = self.dataset_eval_func.single_call(depth_0, filename)
                    except ValueError:
                        import warnings
                        warnings.warn(f"image at sample {filename}  has no usable points")
                        continue
                    errors[camera_type].append(return_dict['error'])
                    abs_errors[camera_type].append(return_dict['abs_error'])

        all_mean_errors = []
        all_mean_errors_abs = []
        for cam in errors:
            mean_errors = np.array(errors[cam]).mean(0)
            mean_abs_errors = np.array(abs_errors[cam]).mean(0)
            self.dataset_eval_func.log(writer, cam, mean_errors, mean_abs_errors, global_step=global_step, epoch_num=epoch_num)
            all_mean_errors.append(mean_errors)
            all_mean_errors_abs.append(mean_abs_errors)
        all_mean_errors = np.array(all_mean_errors).mean(0)
        all_mean_errors_abs = np.array(all_mean_errors_abs).mean(0)
        self.dataset_eval_func.log(writer, 'all mean', all_mean_errors, all_mean_errors_abs, global_step=global_step, epoch_num=epoch_num)

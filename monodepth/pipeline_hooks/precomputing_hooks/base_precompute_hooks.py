import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from vision_base.utils.builder import build


def skew(T):
    return np.array(
        [[0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]]
    )

class BasePrecomputeHook(object):
    """
        Precomputing functions do not have input/output arguments.
        But they can have initialization parameters.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

class MotionMaskPrecomputeHook(BasePrecomputeHook):
    def __init__(self,
                 train_dataset_cfg,
                 flow_estimator_cfg,
                 distance_threshold=5.0,
                 output_dir=''):
        self.dataset = build(**train_dataset_cfg)
        self.flow_estimator_cfg = flow_estimator_cfg
        self.distance_threshold = distance_threshold
        self.output_dir = output_dir

    def __call__(self, *args, **kwargs):
        print(f"Start Precomputing")
        for index in tqdm(range(len(self.dataset)), dynamic_ncols=True):
            target_path = os.path.join(self.output_dir, f"{index:08d}.png")
            if os.path.isfile(target_path):
                continue
            
            data = self.dataset[index]

            image0 = data[("image", 0)]
            image1 = data[("image", 1)]

            gray_image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(gray_image0, gray_image1,
                            None, **self.flow_estimator_cfg)

            H, W, _ = image0.shape
            flow = torch.from_numpy(flow).float().cuda()
            xx_range = torch.arange(0, W).cuda()
            yy_range = torch.arange(0, H).cuda()
            grid_y, grid_x = torch.meshgrid(yy_range, xx_range)
            grid = torch.stack([grid_x, grid_y], dim=-1)

            flowed_grid = grid + flow

            relative_pose = data[('relative_pose', 1)]
            P2 = data['P2']
            R = relative_pose[0:3, 0:3]
            T = relative_pose[0:3, 3]
            T_cross = skew(T)

            K1 = P2[0:3, 0:3]
            K_1 = np.linalg.inv(K1)
            Fundamental = np.transpose(K_1) @ T_cross @ R @ K_1
            Fundamental = torch.from_numpy(Fundamental).float().cuda()

            homo_grid = torch.cat([grid, torch.ones([H, W, 1]).cuda()], dim=-1)
            homo_flowed_grid = torch.cat([flowed_grid, torch.ones([H, W, 1]).cuda()], dim=-1)

            correlations = Fundamental @ homo_grid.reshape(-1, 3).transpose(1, 0)
            correlations = correlations.transpose(1, 0).reshape(H, W, -1)
            denominators = torch.norm(correlations[..., 0:2], dim=-1)

            distances = torch.sum(homo_flowed_grid * (correlations / denominators[..., None]), dim=-1)

            motion_mask = torch.abs(distances) > self.distance_threshold

            cv2.imwrite(os.path.join(self.output_dir, f"{index:08d}.png"),
                motion_mask.cpu().numpy().astype(np.uint8))

            
class MotionMaskARFlowPrecomputeHook(BasePrecomputeHook):
    def __init__(self,
                 train_dataset_cfg,
                 flow_estimator_cfg,
                 distance_threshold=5.0,
                 output_dir=''):
        self.dataset = build(**train_dataset_cfg)
        self.flow_estimator_cfg = flow_estimator_cfg
        self.distance_threshold = distance_threshold
        self.output_dir = output_dir

    def __call__(self, *args, **kwargs):
        print(f"Start Precomputing")
        for index in tqdm(range(len(self.dataset)), dynamic_ncols=True):
            target_path = os.path.join(self.output_dir, f"{index:08d}.png")
            #if os.path.isfile(target_path):
            #    continue
            
            data = self.dataset[index]

            image0 = data[("image", 0)]
            
            flow = data["flow"]

            H, W, _ = image0.shape
            flow = torch.from_numpy(flow).float().cuda()
            flow_norm = torch.norm(flow, dim=-1)
            xx_range = torch.arange(0, W).cuda()
            yy_range = torch.arange(0, H).cuda()
            grid_y, grid_x = torch.meshgrid(yy_range, xx_range)
            grid = torch.stack([grid_x, grid_y], dim=-1)

            flowed_grid = grid + flow

            relative_pose = data[('relative_pose', 1)]
            P2 = data['original_P2']
            R = relative_pose[0:3, 0:3]
            T = relative_pose[0:3, 3]
            T_cross = skew(T)

            K1 = P2[0:3, 0:3]
            K_1 = np.linalg.inv(K1)
            Fundamental = np.transpose(K_1) @ T_cross @ R @ K_1
            Fundamental = torch.from_numpy(Fundamental).float().cuda()

            homo_grid = torch.cat([grid, torch.ones([H, W, 1]).cuda()], dim=-1)
            homo_flowed_grid = torch.cat([flowed_grid, torch.ones([H, W, 1]).cuda()], dim=-1)

            correlations = Fundamental @ homo_grid.reshape(-1, 3).transpose(1, 0)
            correlations = correlations.transpose(1, 0).reshape(H, W, -1)
            denominators = torch.norm(correlations[..., 0:2], dim=-1)

            distances = torch.sum(homo_flowed_grid * (correlations / denominators[..., None]), dim=-1)

            motion_mask = (torch.abs(distances) / flow_norm) > self.distance_threshold

            cv2.imwrite(target_path,
                motion_mask.cpu().numpy().astype(np.uint8))

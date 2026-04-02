import sys
sys.path.append("/home/FSNet")
import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm

from vision_base.utils.builder import build
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.utils.utils import cfg_from_file
from monodepth.networks.models.heads.occupancy_head import project_on_image

print('CUDA available: {}'.format(torch.cuda.is_available()))


def get_coordinates(x_range, y_range, z_range):
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(x_range[0], x_range[1], x_range[2]),
        torch.arange(y_range[0], y_range[1], y_range[2]),
        torch.arange(z_range[0], z_range[1], z_range[2]),
    ), dim=-1).cuda()# [X, Y, Z, 3]
    return coordinates

def get_teacher_voxel(teacher_depth_map, segmentation, z_range, pass_through_loglike=-0.5, hit_loglike=2):
    """
    Compute local log-likelihood voxel from depth maps.
    Args:
        teacher_depth_map: [B, 1, H, W] depth maps.
        segmentation: [B, H, W] long of semantic labels.
        z_range: List in [min_depth, max_depth, depth_interval]
        pass_through_loglike: log-likelihood for voxels that are pass_through by ray casting.
        hit_loglike: log-likelihood for voxels that are hit by any depth.
    Returns:
        target_logits: [B, Z, H, W] log-likelihood voxel.
        semantic_counter: [B, Z, H, W] classification counter
    """
    target_depth_clamp_to_bins = ((teacher_depth_map - z_range[0]) / z_range[2]).long() * z_range[2] + z_range[0]
    depth_max_larger = target_depth_clamp_to_bins + z_range[0]
    depth_bins = torch.arange(z_range[0], z_range[1], z_range[2], device=teacher_depth_map.device) # [Z]
    Z = len(depth_bins)

    reshaped_depth_bins = depth_bins.reshape(1, -1, 1, 1) # [1, Z, 1, 1]
    hit_index = torch.isclose(reshaped_depth_bins, target_depth_clamp_to_bins) # [B, Z, H, W]
    larger_than_max  = (reshaped_depth_bins > depth_max_larger) # [B, Z, H, W]

    semantic_counter = segmentation.unsqueeze(1).repeat(1, Z, 1, 1) #[B, Z, H, W]
    non_hit_mask = torch.logical_not(hit_index)
    semantic_counter[non_hit_mask] = 0 # [B, Z, H, W]

    B, _, H, W = teacher_depth_map.shape
    Z = len(depth_bins)
    target_logits = torch.zeros([B, Z, H, W], device=teacher_depth_map.device) + pass_through_loglike
    target_logits[hit_index] = hit_loglike
    target_logits[larger_than_max] = 0

    return target_logits, semantic_counter


def main():
    cfg = cfg_from_file("configs/occupancy.py")

    split_to_test='training'
    cfg.train_dataset.augmentation = cfg.val_dataset.augmentation
    if split_to_test == 'training':
        dataset = build(**cfg.train_dataset)
    elif split_to_test == 'test':
        dataset = build(**cfg.test_dataset)
    else:
        dataset = build(**cfg.val_dataset)


    teacher_net = build(**cfg.meta_arch.teacher_net_cfg)
    teacher_net.load_state_dict(
        torch.load(cfg.meta_arch.teacher_net_path, map_location='cpu'),
        strict=False
    )
    teacher_net = teacher_net.cuda()

    for param in teacher_net.parameters():
        param.requires_grad=False

    for data in tqdm.tqdm(dataset):
        collated_data = collate_fn([data]) # should be done by dataloader

        for key in collated_data:
            if isinstance(collated_data[key], torch.Tensor):
                collated_data[key] = collated_data[key].cuda().contiguous()
                            
        images = [collated_data[('image', idx)] for idx in cfg.data.frame_idxs]
        P2 = collated_data['P2'] #[B, 3, 4]
        B, _, H, W = images[0].shape
        with torch.no_grad():
            depths = [teacher_net.compute_teacher_depth(collated_data[('image', idx)])[('teacher_depth', 0, 0)] for idx in cfg.data.frame_idxs]
            # List of  [B, 1, H, W]
        teacher_z_range = [2, 50, 0.2] #cfg.meta_arch.head_cfg.z_range, we could use a larger range of depth_bin to get more accurate occupancy
        teacher_depth_bins = torch.arange(teacher_z_range[0], teacher_z_range[1], teacher_z_range[2]) # [Z]
        teacher_Z = len(teacher_depth_bins)

        # ## get teacher voxel
        # teacher_voxels = [get_teacher_voxel(depth, teacher_z_range) for depth in depths] # List of [B, Z, H, W]

        ## get local world 3d voxel
        x_range = cfg.meta_arch.head_cfg.x_range
        y_range = cfg.meta_arch.head_cfg.y_range
        z_range = cfg.meta_arch.head_cfg.z_range
        local_world_voxel = get_coordinates(x_range, y_range, z_range) # [X, Y, Z, 3]
        X, Y, Z, _ = local_world_voxel.shape

        ##
        log_likelihood_occupancies = []
        num_classes=45
        segmentation_counts = torch.zeros([B, X, Y, Z, num_classes]).cuda()
        collated_data[('relative_pose', 0)] = torch.eye(4)[None].repeat([B, 1, 1]).cuda()
        for i, idx in enumerate(cfg.data.frame_idxs):
            T_target2src = collated_data[('relative_pose', idx)] #[B, 4, 4]
            P_T = torch.matmul(P2, T_target2src)
            image_coordinates = project_on_image(local_world_voxel.unsqueeze(0), P_T) # [B, X, Y, Z, 3]
            X_index = image_coordinates[..., 0]
            Y_index = image_coordinates[..., 1]
            Z_index = (image_coordinates[..., 2] - teacher_z_range[0]) / teacher_z_range[2] # [B, X, Y, Z]
            normed_x_index = X_index / W * 2 - 1 #[B, X, Y, Z]
            normed_y_index = Y_index / H * 2 - 1
            normed_z_index = Z_index / teacher_Z * 2 - 1
            if idx == 0:
                pos_weight = 10
                neg_weight = -0.5
            else:
                pos_weight = np.exp(-abs(idx)/10)
                neg_weight = - np.exp(-abs(idx)/10) * 0.5
            segmentation = collated_data[('semantics', idx)] # tensor [B, H, W]
            voxel, segmentation_counter = get_teacher_voxel(depths[i], segmentation, teacher_z_range, hit_loglike=pos_weight, pass_through_loglike=neg_weight) # [B, Z, H, W]
            log_likelihood_occupancies.append(F.grid_sample(
                voxel.unsqueeze(1), #[B, 1, Z, H, W]
                torch.stack([normed_x_index, normed_y_index, normed_z_index], dim=-1), #[B, X, Y, Z, 3]
            )) # [B, 1, X, Y, Z]
            semantic_voxel = F.grid_sample(
                    segmentation_counter.unsqueeze(1).float(), #[B  Z, H, W]
                    torch.stack([normed_x_index, normed_y_index, normed_z_index], dim=-1), #[B, X, Y, Z, 3]
                    mode='nearest',
                ).squeeze(1)  # [B, X, Y, Z]
            semantic_onehot = torch.nn.functional.one_hot(semantic_voxel.long(), num_classes=num_classes)
            semantic_onehot[semantic_voxel==0] = 0
            segmentation_counts = semantic_onehot + segmentation_counts
        log_likelihood_occupancies = torch.cat(log_likelihood_occupancies, dim=1) # [B, T, X, Y, Z]
        target_occupancies = torch.sum(log_likelihood_occupancies, dim=1, keepdim=True) # [B, 1, X, Y, Z]

        value, segmentation_label = torch.max(segmentation_counts, dim=-1, keepdim=False) #[B, X, Y, Z]

        target_occupancies = target_occupancies[0, 0] #[X, Y, Z]
        mask = torch.logical_not(target_occupancies == 0) #[X, Y, Z]
        target_occupancies[target_occupancies > 0] = 1
        target_occupancies[target_occupancies < 0] = 0

        target_occupancies_array = target_occupancies.bool().cpu().numpy()
        mask = mask.bool().cpu().numpy()

        sequence_name = data['sequence_name']
        image_index = data['image_index']

        base_dir = "/home/kitti360_voxel"
        os.makedirs(os.path.join(base_dir, "data_voxel", sequence_name, "image_00"), exist_ok=True)
        output_dir = os.path.join(base_dir, "data_voxel", sequence_name, "image_00", f"{image_index:010d}.npz")
        np.savez_compressed(
            output_dir, occupancy=target_occupancies_array, mask=mask, semantics=segmentation_label[0].cpu().numpy()
        )


if __name__ == "__main__":
    main()

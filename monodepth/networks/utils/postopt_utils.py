import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from monodepth.data.datasets.kitti360_dataset import KITTI360MonoDataset

def denorm(image, rgb_mean, rgb_std):
    new_image = np.clip((image * rgb_std + rgb_mean) * 255, 0, 255)
    new_image = np.array(new_image, dtype=np.uint8)
    return new_image

def _leftcam2imgplane(pts, P2):
    '''
    project the pts from the left camera frame to left camera plane
    pixels = P2 @ pts_cam
    inputs:
        pts(np.array): [#pts, 3]
        points in the left camera frame
    '''
    pixels_T = P2 @ pts.T #(3, #pts)
    pixels = pixels_T.T
    pixels[:, 0] /= pixels[:, 2] + 1e-6
    pixels[:, 1] /= pixels[:, 2] + 1e-6
    return pixels[:, :2]

def read_sparse_vo(dataset, index, output_h, output_w, vo_folder=None):
    instance = dataset.imdb[index]
    
    if isinstance(dataset, KITTI360MonoDataset):
        sequence_name = instance['sequence_name']
        img_index = instance['img_indexes'][0]
        vo_folder = '/data/KITTI-360/sfm_depth_png' if vo_folder is None else vo_folder
        image_path = os.path.join(
            vo_folder, sequence_name, f"{img_index:010d}.png"
        )
    else:
        folder = instance['folder']
        frameindex = instance['index']
        sequence = folder.split('/')[1]
        vo_folder = '/data/kitti_depth_sfm/sfm_depth_png' if vo_folder is None else vo_folder
        image_path = os.path.join(
            vo_folder, sequence, f"{frameindex:010d}.png"
        )
    depth_image = cv2.imread(image_path, -1)
    #if depth_image is None:
    #    print(f"No image found at {image_path}")
    depth_image = cv2.resize(depth_image, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    depth_image_float = depth_image / 65535.0 * 120
    depth_image_float[depth_image_float < 3] = 120
    depth_image_float[depth_image_float > 80] = 120
    return depth_image_float

def read_sparse_depth(dataset, index, image_h=384, image_w=1280, output_h=384, output_w=1280, subsample_ratio=None):
    instance = dataset.imdb[index]
    folder = instance['folder']
    frameindex = instance['index']
    datetime = instance['datetime']
    
    lidar_folder = 'velodyne_points'
    bin_path = os.path.join(dataset.raw_path, folder, lidar_folder, 'data', '%010d.bin' % frameindex)
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts = p[:, 0:3] # [N, 3]
    if subsample_ratio is not None:
        N = len(pts)
        pts = pts[np.random.rand(N) < subsample_ratio,:]
    
    T_vel2cam = dataset.meta_dict[datetime]['T_vel2cam']
    #calib_file = (os.path.join(dataset.raw_path, date_time, "calib_cam_to_cam.txt"))
    
    hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
    pts_hT = np.hstack((pts, hfiller)).T #(4, #pts)
    pts_cam_T = T_vel2cam @ pts_hT # (4, #pts)
    pts_cam = pts_cam_T.T # (#pts, 4)
    P2 = dataset.meta_dict[datetime]['P2']
    pts_2d = _leftcam2imgplane(pts_cam, P2)
    width = image_w
    height = image_h
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pts_cam[:, 2] > 2)
    
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = pts_cam[fov_inds, :]
    depth_map = np.ones((height, width)) * 1e9
    imgfov_pts_2d = imgfov_pts_2d.astype(np.int32)#np.round(imgfov_pts_2d).astype(int)
    
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    depth_map = cv2.resize(depth_map, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return imgfov_pc_rect, depth_map

def depth_image_to_point_cloud_array(depth_image):
    """  convert depth image into color pointclouds [xyzbgr]
    
    """
    w_range = np.arange(0, depth_image.shape[1], dtype=np.float32)
    h_range = np.arange(0, depth_image.shape[0], dtype=np.float32)
    w_grid, h_grid = np.meshgrid(w_range, h_range) #[H, W]

    return np.stack([w_grid, h_grid, depth_image], axis=2)

def SLIC(image, depth_image, h_seg, w_seg, lab_dist_weight=1, iter_num=5, depth_dist_weight=1, image_dist_weight=1):
    H, W, _ = image.shape
    from skimage import color
    lab_arr = color.rgb2lab(image)
    rgb_tensor = torch.from_numpy(lab_arr).permute(2, 0, 1).cuda().unsqueeze(0).float() #[1, 3, H, W]
    cluster_centers_pixel = np.stack(np.meshgrid(
        np.arange(-1, 1.0, 2.0/h_seg), np.arange(-1, 1.0, 2.0/w_seg), indexing='ij'
    ), axis=-1).reshape(-1, 2)
    #cluster_centers_pixel = np.random.rand(h_seg * w_seg, 2) * 2 - 1
    num_segments = len(cluster_centers_pixel)
    center_tensor = torch.from_numpy(cluster_centers_pixel).cuda().float().reshape(1, num_segments, 1, 2).contiguous()
    projected_depth_map_t = torch.from_numpy(depth_image).cuda().permute(2, 0, 1).unsqueeze(0).float().contiguous()

    center_rgb = torch.nn.functional.grid_sample(rgb_tensor, center_tensor, align_corners=True) #[1, 3, 10, 1]
    center_3d = torch.nn.functional.grid_sample(projected_depth_map_t, center_tensor, align_corners=True) #[1, 3, 10, 1]
    
    range_index = torch.arange(num_segments).reshape(num_segments, 1, 1).cuda().contiguous()
    for i in range(iter_num):
        # assign
        rgb_distance = torch.norm(rgb_tensor.unsqueeze(2) - center_rgb.unsqueeze(3), dim=1) #[1, 3, 10, H, W] -> [1, 10, H, W]
        depth_distance = torch.norm(projected_depth_map_t.unsqueeze(2) - center_3d.unsqueeze(3), dim=1) #[1, 3, 10, H, W] -> [1, 10, H, W]
        diff = projected_depth_map_t.unsqueeze(2) - center_3d.unsqueeze(3) #[1, 3, 10, H, W]
        image_distance = torch.norm(diff[:, 0:2], dim=1)
        depth_distance = torch.abs(diff[:, 2])
        
        total_loss = rgb_distance  * lab_dist_weight +\
                     depth_distance * depth_dist_weight +\
                     image_distance * image_dist_weight
        _, min_index = torch.min(total_loss, dim=1) #[1, H, W]
        combined_selection_mask = (range_index == min_index) #[10, H, W]
        # Update
        total_number_each_center = torch.sum(combined_selection_mask, dim=[1, 2], keepdim=True) + 1e-4 #[10, 1, 1]
        new_mean_rgb = torch.sum(rgb_tensor.unsqueeze(2) * combined_selection_mask, dim=[-1, -2], keepdim=True) / total_number_each_center
        center_rgb = new_mean_rgb.squeeze(-1) #[1, 3, 10, 1]
        new_mean_3d = torch.sum(projected_depth_map_t.unsqueeze(2) * combined_selection_mask, dim=[-1, -2], keepdim=True) / total_number_each_center
        if torch.all(new_mean_3d.squeeze(-1) == center_3d):
            print(f"Break at iter {i}")
            break
        center_3d  = new_mean_3d.squeeze(-1)  #[1, 3, 10, 1]
    
    segments = []
    centers = []
    for i in range(num_segments):
        segment = torch.nonzero(combined_selection_mask[i], as_tuple=False)
        if len(segment) > 0:
            segments.append(
                 segment # .cpu()
            ) #[N_i, 2]
            centers.append(center_3d[0, 0:2, i, 0])
    
    return torch.stack(centers, dim=1) , segments #[3, N], #[2, N]

def select_best_vo_points(log_pred, log_vo, max_points):
    H, W = log_pred.shape
    reshaped_log_pred = log_pred.reshape(-1) #[H*W]
    reshaped_log_vo = log_vo.reshape(-1) #[H*W]
    base_valid_mask = (reshaped_log_vo < np.log(80)) * (reshaped_log_vo > np.log(3)) #[H*W]
    if base_valid_mask.sum() < max_points:
        return base_valid_mask.reshape(H, W)
    diff = reshaped_log_pred - reshaped_log_vo #[H*W]
    _, top_indices = torch.topk(diff.abs(), max_points, sorted=False, largest=False)
    top_k_mask = torch.zeros_like(base_valid_mask)
    top_k_mask[top_indices] = 1
    valid_mask = (base_valid_mask * top_k_mask).reshape(H, W)
    return valid_mask

def post_optimization(
    image, depth_image, depth_prediction, reference_depth, h_seg, w_seg,
    lab_dist_weight=1, iter_num=5, depth_dist_weight=1, image_dist_weight=1,
    lambda0 = 0.000,
    lambda1 = 1.0,
    lambda2 = 0.001,
    max_distance=100,
    max_points=800,
):
    centers, segments = SLIC(image, depth_image, lab_dist_weight=lab_dist_weight,
                     h_seg=h_seg, w_seg=w_seg,
                     iter_num=iter_num,
                     depth_dist_weight=depth_dist_weight,
                     image_dist_weight=image_dist_weight)

    log_pred = torch.log(depth_prediction).cuda().float()
    if isinstance(reference_depth, np.ndarray):
        reference_depth = torch.from_numpy(reference_depth)
    log_point = torch.log(reference_depth).cuda().float()

    ## Select closest K points
    valid_mask = select_best_vo_points(log_pred, log_point, max_points=max_points)

    base_scales = torch.ones(len(segments)).cuda()
    target_scales = torch.ones(len(segments)).cuda()
    lambda1_mask = torch.zeros(len(segments)).cuda()

    for i, segment in enumerate(segments):
        log_depth_base = log_pred[segment[:, 0], segment[:, 1]] #[N, ]
        sparse_depth_segment   = log_point[segment[:, 0], segment[:, 1]] #[N, ]

        valid = valid_mask[segment[:, 0], segment[:, 1]]
        base_scales[i] = torch.mean(log_depth_base)
        if valid.sum() < 1:
            lambda1_mask[i] = 0
        else:
            lambda1_mask[i] = 1
            target_scales[i] = (sparse_depth_segment[valid] - log_depth_base[valid]).mean() + base_scales[i]
    roki = base_scales[:, None] - base_scales[None, :]
    center_diff = torch.norm(
        centers[:, None, :] - centers[..., None], dim=0 #[3, 1, N] - [3, N, 1] = [3, N, N]
        ) # [N, N]
    weights = torch.exp(-center_diff/20)
    sum_weights = torch.sum(weights, dim=-1)

    lambda1_array = lambda1 * lambda1_mask
    A = torch.diag(sum_weights * lambda0 + lambda1_array + lambda2) - lambda0  * weights
    B = lambda2 * base_scales + lambda1_array * target_scales + lambda0 * torch.sum(roki * weights, dim=-1)

    new_scale = torch.matmul(A.inverse(),B.reshape([-1, 1]))# torch.linalg.solve(A, B.reshape([-1, 1])) # torch.matmul(A.inverse(),B )
    new_scale_diff = new_scale[:, 0] - base_scales
    for i, segment in enumerate(segments):
        log_pred[segment[:, 0], segment[:, 1]] += new_scale_diff[i] #[N, ]
    new_depth = torch.exp(log_pred)
    
    
    return new_depth

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(
            self.batch_size, 4, self.height, self.width)

        return cam_points

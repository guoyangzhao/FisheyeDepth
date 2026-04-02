import os
from scipy.spatial.transform import Rotation
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def depth_to_disp(depth, min_depth, max_depth):
    """Convert network's depth prediction into sigmoid output
    """

    disp = (1 / depth - 1 / max_depth) / (1 / min_depth - 1 / max_depth)
    return disp

def inverse_sigmoid(x):
    """Inverse sigmoid function
    """
    return torch.log(x / (1 - x + 1e-8))

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rotation_matrix_to_euler_viascipy(rotation_matrixes:torch.Tensor, axis_sequence:str) -> torch.Tensor:
    """
    Rotation matrix to euler, tensor version, but go through scipy.

    Arguments:
        rotation_matrixes: torch.Tensor shape [B, 3, 3]
        axis_array: str indicates the sequence of euler angle axis

    Returns:
        torch.Tensor [B, 3]
    """
    rotation_numpy = rotation_matrixes.detach().cpu().numpy()
    rot_list = [Rotation.from_matrix(rot) for rot in rotation_numpy] #List[Rotation object]
    zyx_euler_tensor = torch.stack(
        [rotation_matrixes.new(rot.as_euler(axis_sequence).copy()) for rot in rot_list],
        dim=0)
    return zyx_euler_tensor

def gather_activation(x:torch.Tensor, depth_bins:torch.Tensor, min_depth=0.1, max_depth=100) -> torch.Tensor:
    """Decode the output of the cost volume into a encoded depth feature map.

    Args:
        x (torch.Tensor): The output of the cost volume of shape [B, num_depth_bins, H, W]
        depth_bins (torch.Tensor): Depth bins of shape [num_depth_bins]

    Returns:
        torch.Tensor: Encoded depth feature map of shape [B, 1, H, W]
    """
    assert x.dim() == 4, "Activation must be of shape [B, Depth_bin, H, W]"
    assert x.shape[1] == depth_bins.shape[0], "Activation and depth_bins must have the same depth"

    activated = torch.softmax(x, dim=1)
    activation_bins = inverse_sigmoid(depth_to_disp(depth_bins, min_depth, max_depth))
    encoded_depth = torch.sum(activated * activation_bins.reshape(1, -1, 1, 1), dim=1, keepdim=True)
    return encoded_depth

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    @staticmethod
    def get_grid(batch_size, height, width):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = torch.from_numpy(np.stack(meshgrid, axis=0).astype(np.float32))

        ones = torch.ones(batch_size, 1, height * width)

        pix_coords = torch.unsqueeze(torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        homo_pix_coords = torch.cat([pix_coords, ones], 1)

        return pix_coords, homo_pix_coords

    @staticmethod
    def get_normalize_grid(batch_size, height, width):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        meshgrid[0] = meshgrid[0] / width * 2 - 1
        meshgrid[1] = meshgrid[1] / height * 2 - 1
        id_coords = torch.from_numpy(np.stack(meshgrid, axis=0).astype(np.float32)) #[2, H, W]

        pix_coords = id_coords.unsqueeze(0) #[1, 2, H, W]
            
        pix_coords = pix_coords.repeat(batch_size, 1, 1, 1)

        return pix_coords

    def forward(self, depth, inv_K):
        batch, _, height, width = depth.shape

        ones  = inv_K.new_ones([batch, 1, height * width])

        _, homo_pix_coords = self.get_grid(batch, height, width)

        cam_points = torch.matmul(inv_K[:, :3, :3], homo_pix_coords.cuda())
        cam_points = depth.view(batch, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, eps=1e-7):
        super(Project3D, self).__init__()

        self.eps = eps

    def forward(self, points, K, T, batch_size, height, width):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(batch_size, 2, height, width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= width - 1
        pix_coords[..., 1] /= height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, kernel_size=3, padding=1):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(kernel_size, 1)
        self.mu_y_pool   = nn.AvgPool2d(kernel_size, 1)
        self.sig_x_pool  = nn.AvgPool2d(kernel_size, 1)
        self.sig_y_pool  = nn.AvgPool2d(kernel_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(kernel_size, 1)

        self.refl = nn.ReflectionPad2d(padding)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.kernel_size=kernel_size

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class SSIMUncer(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def forward(self, x0, y0):
        x = self.refl(x0)
        y = self.refl(y0)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        N = self.kernel_size ** 2
        dsigma_y_dy = 2.0 / N * y0 - 2 / N * mu_y
        dsigma_xy_dy = 1 / N * x0 - 1 / N * mu_x

        Id = mu_x ** 2 + mu_y ** 2 + self.C1
        I = (2 * mu_x * mu_y + self.C1) / Id # noqa: E741

        Cs_n = (2 * sigma_xy + self.C2)
        Cs_d = (sigma_x + sigma_y + self.C2)
        Cs = Cs_n / Cs_d

        dIdy = (2 * mu_x * (mu_x ** 2 - mu_y ** 2) ) / (Id ** 2)
        dCsdy = (Cs_d * (2 * dsigma_xy_dy) - Cs_n * (dsigma_y_dy)) / (Cs_d ** 2)

        output = torch.clamp((1 - I * Cs) / 2, 0, 1)
        doutput_dy = - (I * dCsdy + Cs * dIdy) / 2

        return output, doutput_dy


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1] # type: ignore
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

def project_depth_map(velo:np.ndarray, P_velo2im:np.ndarray, im_shape:np.ndarray):
    """Generate a depth map from velodyne data
    """
    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo_input = velo[velo[:, 0] >= 0, :].copy()
    velo_input[:, 3] = 1.0  # homogeneous

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo_input.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    velo_pts_im[:, 2] = velo_input[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1] # type: ignore
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


def decode_depth_inv_sigmoid(depth:torch.Tensor)->torch.Tensor:
    """Decode depth from network prediction to 3D depth

    Args:
        depth (torch.Tensor): depth from network prediction (un-activated)

    Returns:
        torch.Tensor: 3D depth for output
    """
    depth_decoded = torch.exp(-depth) #1 / torch.sigmoid(depth) - 1
    return depth_decoded

def encode_depth_inv_sigmoid(depth_decoded:torch.Tensor)->torch.Tensor:
    """Decode depth from network prediction to 3D depth

    Args:
        depth_decoded (torch.Tensor): depth from network prediction (un-activated)

    Returns:
        torch.Tensor: 3D depth for output
    """
    if isinstance(depth_decoded, torch.Tensor):
        depth = - torch.log(depth_decoded)
    if isinstance(depth_decoded, np.ndarray):
        depth = - np.log(depth_decoded)
    return depth

def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)

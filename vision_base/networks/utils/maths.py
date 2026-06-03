from scipy.spatial.transform import Rotation
import numpy as np
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

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
        meshgrid[0] = meshgrid[0] / height * 2 - 1
        meshgrid[1] = meshgrid[1] / width * 2 - 1
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


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)

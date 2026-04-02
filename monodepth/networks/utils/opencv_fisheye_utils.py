import torch
import numpy as np
from numba import jit

"""
    OpenCV fisheye camera model is presented at:
    https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html

    Following the need of our project, we expect the input images are undistorted.

    1. The pinhole projection on middle numbers, a = x/z and b = y/z, r^2 = a^2 + b^2, \theta = atan(r).
    1.5 Distortion, \theta = \theta * (1 + k1 * \theta^2 + k2 * \theta^4 + k3 * \theta^6 + k4 * \theta^8)  (ignored)
    2. The distorted point coordinate are x' = (\theta / r) * a , y' = (\theta / r) * b.
    3. The final image coordinate are x = x' * f + cx, y = y' * f + cy.
"""

def _cam2image(points, P, calib):
    """camera coordinate to image plane, input is array/tensor of [xxxx, 3]"""
    if isinstance(points, np.ndarray):
        norm = np.linalg.norm(points, axis=-1)
        abs_func = np.abs
        sqrt = np.sqrt
        arctan = np.arctan
    elif isinstance(points, torch.Tensor):
        norm = torch.norm(points, dim=-1)
        abs_func = torch.abs
        sqrt = torch.sqrt
        arctan = torch.atan
    else:
        raise NotImplementedError

    eps = 1e-6
    a = points[..., 0] / (points[..., 2] + eps)
    b = points[..., 1] / (points[..., 2] + eps)
    r = sqrt(a * a + b * b)
    theta = arctan(r)

    dist = calib['distortion_parameters']
    theta = theta * (1 + dist['k1'] * theta**2 + dist['k2'] * theta**4 + dist['k3'] * theta**6 + dist['k4'] * theta**8)

    x1 = (theta / r) * a
    y1 = (theta / r) * b

    gamma1 = P[0, 0]
    gamma2 = P[1, 1]
    u0 = P[0, 2]
    v0 = P[1, 2]
    x = gamma1 * x1 + u0
    y = gamma2 * y1 + v0

    return x, y, norm * points[..., 2] / (abs_func(points[..., 2]) + eps)


"""
    Reproject image plane to camera coordinate based on the OpenCV camera model.

    The inverse computation contains the following steps:
    1. retrieve points from the image plane to the normalized plane: x' = (x - u0) / gamma1; y' = (y - v0) / gamma2
    2. compute \theta = sqrt(x' * x' + y' * y'), r = tan(\theta)
    3. compute a = r * x' / \theta, b = r * y' / \theta
    4. z = norm / (sqrt(a*a + b*b + 1)), x = a * z, y = b * z
"""

@jit(nopython=True, cache=True)
def radial_distort_func(k1, k2, k3, k4, r1, r0):
    return r0 - r1 / (1 + k1 * r0**2 + k2 * r0**4 + k3 * r0**6 + k4 * r0**8)

@jit(nopython=True, cache=True)
def newton_methods(x0, k1, k2, k3, k4, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        f = radial_distort_func(k1, k2, k3, k4, x0, x)
        if abs(f) < tol:
            return x
        df = (radial_distort_func(k1, k2, k3, k4, x0, x + tol) - f) / tol
        x = x - f / df
    return x


@jit(nopython=True, cache=True)
def whole_map_undistort(H, W, theta, k1, k2, k3, k4):
    theta2 = np.ones((1, 1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            theta2[0, 0, i,j] = newton_methods(theta[0, 0, i, j], k1, k2, k3, k4)
    
    return theta2

class OpenCVFisheyeCameraProjection(object):
    def __init__(self):
        self.cache = {}
    

    def get_grid(self, height, width):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        pix_coords =np.stack([id_coords[0], id_coords[1]], 0)[None]

        return pix_coords

    def cam2image(self, points, P, calib):
        x, y, norm = _cam2image(points, P, calib)
        return torch.stack([x, y, norm], dim=-1)

    def image2cam(self, norm, P, calib):
        _, _ ,H, W = norm.shape
        
        u0 = P[:, 0, 2] # [B]
        v0 = P[:, 1, 2]
        gamma1 = P[:, 0, 0]
        gamma2 = P[:, 1, 1]

        ## get grid
        xy = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        meshgrid = [torch.from_numpy(array).float().to(norm.device) for array in xy]
        # meshgrid = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy').to(norm.device)
        id_coords = torch.stack(meshgrid, dim=0).float()
        xy_grid = torch.stack([id_coords[0], id_coords[1]], dim=0)[None] #[1, 2, H, W]

        ## normalized grid
        X = (xy_grid[:, 0:1] - u0[:, None, None, None]) / gamma1[:, None, None, None] # [B, 1, H, W]
        Y = (xy_grid[:, 1:2] - v0[:, None, None, None]) / gamma2[:, None, None, None] # [B, 1, H, W]
        
        ## back track fisheye
        eps = 1e-6
        theta = torch.sqrt(X**2 + Y**2) # [B, 1, H, W]
        undist_theta = theta.clone()
        for b in range(len(theta)):
            dist = calib[b]['distortion_parameters']
            k1, k2, k3, k4 = dist['k1'], dist['k2'], dist['k3'], dist['k4']
            key = (H, W, gamma1[b].item(), gamma2[b].item(), u0[b].item(), v0[b].item(), k1, k2, k3, k4)
            if key in self.cache:
                undist_theta[b:b+1] = torch.from_numpy(self.cache[key]).float().cuda()
            else:
                undist_theta[b:b+1] = torch.from_numpy(whole_map_undistort(H, W, theta[b:b+1].cpu().numpy(), k1, k2, k3, k4)).float().cuda()
                self.cache[key] = undist_theta[b:b+1].cpu().numpy().copy()
        r = torch.tan(undist_theta) # [B, 1, H, W]
        a = r * X / (theta + eps) # [B, 1, H, W]
        b = r * Y / (theta + eps) # [B, 1, H, W]
        angles = torch.sqrt(a**2 + b**2 + 1)
        mask = (angles < 20) * (r > 0)

        ## compute z, x, y
        z = norm / angles # [B, 1, H, W]
        x = a * z # [B, 1, H, W]
        y = b * z # [B, 1, H, W]

        return torch.stack([x, y, z], dim=-1), mask

import torch
import numpy as np
from numba import jit

"""
    Mei unified camera model is presented at:
    https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf

    1. Project the points in the camera frame into a unit sphere.  X = X / norm
    2. Project the point onto a new normalized plane define by the mirror parameters. X = X / (Z + xi); Y = Y / (Z + xi)
    3. Apply the distortion model to the normalized plane. X = X * (1 + k1 * ro2 + k2 * ro2 * ro2); Y = Y * (1 + k1 * ro2 + k2 * ro2 * ro2)
    4. Project the point onto the image plane. X = gamma1 * X + u0; Y = gamma2 * Y + v0
"""
def mei_distort(normalized_x, normalized_y, calib):
    k1 = calib["distortion_parameters"]["k1"]
    k2 = calib["distortion_parameters"]["k2"]

    ro2 = normalized_x * normalized_x + normalized_y * normalized_y
    x = normalized_x * (1 + k1 * ro2 + k2 * ro2 * ro2)
    y = normalized_y * (1 + k1 * ro2 + k2 * ro2 * ro2)
    return x, y

def _cam2image(points, P, calib):
    """camera coordinate to image plane, input is array/tensor of [xxxx, 3]"""
    if isinstance(points, np.ndarray):
        norm = np.linalg.norm(points, axis=-1)
        abs_func = np.abs
    elif isinstance(points, torch.Tensor):
        norm = torch.norm(points, dim=-1)
        abs_func = torch.abs
    else:
        raise NotImplementedError

    eps = 1e-6
    x = points[..., 0] / (norm + eps)
    y = points[..., 1] / (norm + eps)
    z = points[..., 2] / (norm + eps)

    x /= z + calib["mirror_parameters"]["xi"] + eps
    y /= z + calib["mirror_parameters"]["xi"] + eps

    x, y = mei_distort(x, y, calib)

    gamma1 = P[0, 0]
    gamma2 = P[1, 1]
    u0 = P[0, 2]
    v0 = P[1, 2]
    x = gamma1 * x + u0
    y = gamma2 * y + v0

    return x, y, norm * points[..., 2] / (abs_func(points[..., 2]) + eps)


"""
    Reproject image plane to camera coordinate based on the Mei camera model.

    Reprojecting the image plane to the camera coordinate is a little bit tricky, which requires to solve multiple non-linear equations. Since we have a rather stable camera model, we can use the iterative method to solve the equations and cache the results.

    The inverse computation contains the following steps:
    1. retrieve points from the image plane to the normalized plane: X = (x - u0) / gamma1; Y = (y - v0) / gamma2
    2. Backtracking the radial distortion model, we have an forward equation r1 = r0(1 + k1 * r0^2 + k2 * r0^4) and a backward equation r0 = r1 / (1 + k1 * r1^2 + k2 * r1^4). We can use the Newton method to solve the equation. Then we backtrack the normalized plane coordinates: X = X * r0 / r1; Y = Y * r0 / r1. We will cache the result in an image because it is a constant value for each pixel.
    3. Backtracking the mirror parameters, we have a forward equation r0^2 = (1 - Z^2) / (xi + Z)^2. This function is monotonic decreasing given Z in [0, 1]. So we use bisection method to solve the equation. We will cache the result in an image because it is a constant value for each pixel.
    4. Backtracking the camera coordintate. Given we have z from network prediction. norm = Z / z. We can compute the camera coordinate by X3d = X * norm; Y3d = Y * norm;
"""

@jit(nopython=True, cache=True)
def radial_distort_func(k1, k2, r1, r0):
    return r0 - r1 / (1 + k1 * r0**2 + k2 * r0**4)

@jit(nopython=True, cache=True)
def newton_methods(k1, k2, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        f = radial_distort_func(k1, k2, x0, x)
        if abs(f) < tol:
            return x
        df = (radial_distort_func(k1, k2, x0, x + tol) - f) / tol
        x = x - f / df
    return x

@jit(nopython=True, cache=True)
def mirror_backtrack_func(r0, xi, Z):
    return r0**2 - (1 - Z**2) / (xi + Z)**2

@jit(nopython=True, cache=True)
def bisection_methods(r0, xi, x0, x1, tol=1e-6, max_iter=100):
    y0 = mirror_backtrack_func(r0, xi, x0)
    y1 = mirror_backtrack_func(r0, xi, x1)
    if y0*y1 > 0:
        return False, x0-1

    for i in range(max_iter):
        x = (x0 + x1) / 2
        f = mirror_backtrack_func(r0, xi, x)
        if abs(f) < tol:
            return True, x
        if f * mirror_backtrack_func(r0, xi, x0) < 0:
            x1 = x
        else:
            x0 = x
    return True, x

@jit(nopython=True, cache=True)
def backtracking_mei_func(r1, k1, k2, xi):
    ## backtrack radial distortion model
    r0 = newton_methods(k1, k2, r1)

    ## backtrack mirror parameters
    flag, Z = bisection_methods(r0, xi, 0, 1)
    return flag, Z

@jit(nopython=True, cache=True)
def whole_map_backtracking(H, W, r1, k1, k2, xi):
    Z = np.ones((1, 1, H, W), dtype=np.float32)
    mask = np.ones((1, 1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            mask[0, 0, i,j], Z[0, 0, i, j] = backtracking_mei_func(r1[0, 0, i, j], k1, k2, xi)
    
    return mask, Z

class MeiCameraProjection(object):
    def __init__(self):
        self.cache = {}
    

    def get_grid(self, height, width):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        pix_coords =np.stack([id_coords[0], id_coords[1]], 0)[None]

        return pix_coords

    def cam2image(self, points, P, calib):
        x, y, z = _cam2image(points, P, calib)
        if isinstance(points, np.ndarray):
            return np.stack([x, y, z], axis=-1)
        return torch.stack([x, y, z], dim=-1)

    def image2cam(self, norm, P, calib):
        B, _ ,H, W = norm.shape
        Xs = []
        Ys = []
        Zs = []
        masks = []
        for b in range(B):
            k1 = calib[b]["distortion_parameters"]["k1"]
            k2 = calib[b]["distortion_parameters"]["k2"]
            xi = calib[b]["mirror_parameters"]["xi"]
            u0 = P[b, 0, 2].item()
            v0 = P[b, 1, 2].item()
            gamma1 = P[b, 0, 0].item()
            gamma2 = P[b, 1, 1].item()
            if (H, W, gamma1, gamma2, u0, v0, k1, k2, xi) in self.cache:
                X, Y, Z, mask= self.cache[(H, W, gamma1, gamma2, u0, v0, k1, k2, xi)] #[1, 1, H, W]
            else:
                xy_grid = self.get_grid(H, W) #[1, 2, H, W]
                X = (xy_grid[:, 0:1] - u0) / gamma1
                Y = (xy_grid[:, 1:2] - v0) / gamma2 #[1, 1, H, W]
                
                r1 = np.sqrt(X**2 + Y**2)
                mask, Z = whole_map_backtracking(H, W, r1, k1, k2, xi)
                # mask[Z < 0.05] = 0    ##gy##
                not_mask = np.logical_not(mask)
                Z[not_mask] = -1
                X[not_mask] = -1
                Y[not_mask] = -1
                X = X * (Z + xi)
                Y = Y * (Z + xi)
                X = torch.from_numpy(X)
                Y = torch.from_numpy(Y)
                Z = torch.from_numpy(Z)
                
                self.cache[((H, W, gamma1, gamma2, u0, v0, k1, k2, xi))] = (X, Y, Z, mask)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
            masks.append(torch.from_numpy(mask))

        X = torch.cat(Xs, dim=0).to(norm.device)
        Y = torch.cat(Ys, dim=0).to(norm.device)
        Z = torch.cat(Zs, dim=0).to(norm.device)
        mask_tensor = torch.cat(masks, dim=0).to(norm.device)
        #norm = depth / Z
        z = Z * norm
        x = X * norm
        y = Y * norm
        return torch.stack([x, y, z], dim=-1), mask_tensor

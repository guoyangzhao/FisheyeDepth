
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import scipy.io as sio
import cv2

def read_pc_from_bin(bin_path:str)->np.ndarray:
    """Load PointCloud data from bin file."""
    p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return p

def read_vo_depth(image_path):
    depth_image = cv2.imread(image_path, -1)
    #if depth_image is None:
    #    print(f"No image found at {image_path}")
    depth_image_float = depth_image / 65535.0 * 120
    depth_image_float[depth_image_float < 3] = 120
    depth_image_float[depth_image_float > 80] = 120
    return depth_image_float

def read_image(path:str)->np.ndarray:
    '''
    read image
    inputs:
        path(str): image path
    returns:
        img(np.array): [w,h,c] [r, g, b]
    '''
    return np.array(Image.open(path, 'r'))

def read_depth(path:str)->np.ndarray:
    """ Read Ground Truth Depth Image
    
    Args:
        path: image path
    Return:
        depth image: floating image [H, W]
    """
    return np.array((cv2.imread(path, -1)) / 256.0, dtype=np.float32)

def read_pose_mat(path:str) -> np.ndarray:
    """ Read Pose generated from matlab devkit
    
    Args:
        path: mat file path
    Return:
        Poses: float numpy array [N, 4, 4]
    """
    return sio.loadmat(path)['pose_mat']


def cam_relative_pose(T_imu2world_0:np.ndarray, T_imu2world_1:np.ndarray, T_imu2vel:np.ndarray, T_vel2cam:np.ndarray):
    return T_vel2cam @ T_imu2vel @ np.linalg.inv(T_imu2world_1) @ T_imu2world_0 @ np.linalg.inv(T_imu2vel) @ np.linalg.inv(T_vel2cam)

def cam_relative_pose_nusc(T_imu2world_0:np.ndarray, T_imu2world_1:np.ndarray, T_imu2cam:np.ndarray):
    return T_imu2cam @ np.linalg.inv(T_imu2world_1) @ T_imu2world_0 @ np.linalg.inv(T_imu2cam)

def get_transformation_matrix(translation, rotation):
    """ Compute transformation matrix T [4x4] from translation [x, y, z] and quaternion rotation [w, x, y, z]
    """
    rotation = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]) #[x, y, z, w]
    rotation_matrix = rotation.as_matrix() #[3, 3]
    T = np.eye(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = translation
    return T

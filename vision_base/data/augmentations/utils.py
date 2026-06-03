import numpy as np
from scipy.spatial.transform import Rotation as R

def flip_relative_pose(pose:np.ndarray, axis_num=0):
    """ Compute the resulting pose when the image/world is flipped
    """
    rotation = R.from_matrix(pose[0:3, 0:3])
    xyz_rotation = rotation.as_euler('xyz')
    for i in range(3):
        if i != axis_num:
            xyz_rotation[i] = xyz_rotation[i] * -1

    new_rotation = R.from_euler('xyz', xyz_rotation)
    t = pose[0:3, 3:4]
    t[axis_num, :] *= -1

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0:3, 0:3] = new_rotation.as_matrix()
    new_pose[0:3, 3:4] = t
    return new_pose

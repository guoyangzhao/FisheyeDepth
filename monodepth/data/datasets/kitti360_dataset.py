from __future__ import print_function, division
import os
from easydict import EasyDict
import numpy as np
from copy import deepcopy

import torch
import torch.utils.data
import cv2
from torch.utils.data import Dataset, DataLoader # noqa: F401
from monodepth.data.datasets.utils import read_image,  cam_relative_pose_nusc, read_csv
from vision_base.utils.builder import build

def read_P01_from_sequence(file):
    """ read P0 and P1 from a sequence file perspective.txt
    """
    P0 = None
    P1 = None
    R0 = np.eye(4)
    R1 = np.eye(4)
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("P_rect_00"):
                data = line.strip().split(" ")
                P0 = np.array([float(x) for x in data[1:13]])
                P0 = np.reshape(P0, [3, 4])
            if line.startswith("R_rect_00"):
                data = line.strip().split(" ")
                R = np.array([float(x) for x in data[1:10]])
                R0[0:3, 0:3] = np.reshape(R, [3, 3])
            if line.startswith("P_rect_01"):
                data = line.strip().split(" ")
                P1 = np.array([float(x) for x in data[1:13]])
                P1 = np.reshape(P1, [3, 4])
            if line.startswith("R_rect_01"):
                data = line.strip().split(" ")
                R = np.array([float(x) for x in data[1:10]])
                R1[0:3, 0:3] = np.reshape(R, [3, 3])
    return P0, P1, R0, R1

def read_extrinsic_from_sequence(file):

    T0 = np.eye(4)
    T1 = np.eye(4)
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("image_00"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T0[0:3, :] = np.reshape(T, [3, 4])
            if line.startswith("image_01"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T1[0:3, :] = np.reshape(T, [3, 4])
    

    return T0, T1

def read_poses_file(file):
    key_frames = []
    poses = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            key_frames.append(int(data[0]))
            pose = np.eye(4)
            pose[0:3, :] = np.array([float(x) for x in data[1:13]]).reshape([3, 4])
            poses.append(pose)
    poses = np.array(poses)
    return key_frames, poses

def read_T_from_sequence(file):
    """ read T from a sequence file calib_cam_to_velo.txt
    """
    with open(file, 'r') as f:
        line = f.readlines()[0]
        data = line.strip().split(" ")
        T = np.array([float(x) for x in data[0:12]]).reshape([3, 4])

    T_velo2cam = np.eye(4)
    T_velo2cam[0:3, :] = T
    return T_velo2cam

class KITTI360MonoDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(KITTI360MonoDataset, self).__init__()
        self.raw_path = getattr(data_cfg, 'raw_path', '/data/KITTI-360')
        self.meta_file   = getattr(data_cfg, 'split_file', 'kitti360_meta.txt')
        self.read_mask   = getattr(data_cfg, 'read_mask', False)

        self.img_dir   = os.path.join(self.raw_path, 'data_2d_raw')
        self.pose_dir  = os.path.join(self.raw_path, 'data_poses')
        self.calib_dir = os.path.join(self.raw_path, 'calibration')
        self.pc_dir    = os.path.join(self.raw_path, 'data_3d_raw')

        self.frame_ids = getattr(data_cfg, 'frame_ids', [0, -1, 1])
        self.imdb = []
        self.sequence_names = set()

        header, data = read_csv(self.meta_file)
        for data_cell in data:
            sequence_name = data_cell['sequence_name']
            pose_index = int(data_cell['pose_index'])
            pose_indexes = [pose_index + ind for ind in self.frame_ids]
            img_indexes = [int(data_cell[str(ind)]) for ind in self.frame_ids]

            self.sequence_names.add(sequence_name)
            self.imdb.append(
                    dict(
                        sequence_name=sequence_name,
                        pose_indexes=pose_indexes,
                        img_indexes=img_indexes,
                    )
                )
        
        self._load_calib()
        self._load_keypose()

        self.is_motion_mask  = getattr(data_cfg, 'is_motion_mask', False)
        if self.is_motion_mask:
            self.precompute_path = getattr(data_cfg, 'motion_mask_path', "")

        self.is_filter_static = getattr(data_cfg, 'is_filter_static', True)
        self.filter_threshold = getattr(data_cfg, 'filter_threshold', 0.03)
        if self.is_filter_static:
            self.imdb = self._filter_indexes()

        self.use_right_image = getattr(data_cfg, 'use_right_image', True)
        self.image_suffix    = getattr(data_cfg, 'image_suffix', 'png')

        self.transform = build(**data_cfg.augmentation)
    
    def _filter_indexes(self):
        imdb = []
        print(f"Start Filtering indexes, original length {len(self)}")
        for obj in self.imdb:
            is_overlook = False
            sequence_name = obj['sequence_name']
            pose_indexes = obj['pose_indexes']
            extrinsics = self.cam_calib['T_rect02baselink']
            poses = self.keypose[sequence_name][pose_indexes]

            for i, idx in enumerate(self.frame_ids[1:]):
                pose_diff = cam_relative_pose_nusc(
                    poses[0], poses[i+1], np.linalg.inv(extrinsics)
                ).astype(np.float32)
                translation = np.linalg.norm(pose_diff[0:3, 3])
                if translation < self.filter_threshold or translation > 3:
                    is_overlook=True

            if not is_overlook:
                imdb.append(obj)
        print(f"Finished filtering indexes, find dynamic instances {len(imdb)}")
        return imdb

    def _load_calib(self):
        cam_calib_file = os.path.join(self.calib_dir, "perspective.txt")
        cam_extrinsic_file = os.path.join(self.calib_dir, "calib_cam_to_pose.txt")

        P0, P1, R0, R1 = read_P01_from_sequence(cam_calib_file)
        T0, T1 = read_extrinsic_from_sequence(cam_extrinsic_file)
        self.cam_calib = dict()
        self.cam_calib['P0'] = P0
        self.cam_calib['P1'] = P1
        self.cam_calib['T_rect02baselink'] = R0 @ T0
        self.cam_calib['T_rect12baselink'] = R1 @ T1

        if self.read_mask:
            self.masks = []
            for i in range(2):
                image_path = os.path.join(self.raw_path, f'mask{i:02d}.png')
                self.masks.append(cv2.imread(image_path, -1))

    def _load_keypose(self):
        self.keypose = {}
        for sequence_name in self.sequence_names:
            poses_file = os.path.join(self.pose_dir, sequence_name, 'poses.txt')
            key_frames, poses = read_poses_file(poses_file)
            self.keypose[sequence_name] = poses

    def __len__(self):
        return len(self.imdb)
    
    
    def __getitem__(self, index):
        obj = self.imdb[index]
        sequence_name = obj['sequence_name']
        pose_indexes = obj['pose_indexes']
        img_indexes = obj['img_indexes']

        if (not self.use_right_image) or (np.random.rand() < 0.5):
            extrinsics = self.cam_calib['T_rect02baselink']
            image_dir_name = 'image_00'
            P2 = self.cam_calib['P0']
        else:
            extrinsics = self.cam_calib['T_rect12baselink']
            image_dir_name = 'image_01'
            P2 = self.cam_calib['P1']
        
        data = dict()
        poses = self.keypose[sequence_name][pose_indexes] #[3, 4, 4]
        for i, idx in enumerate(self.frame_ids[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(
               poses[0], poses[i+1], np.linalg.inv(extrinsics)
            ).astype(np.float32)
        
        image_dir = os.path.join(self.img_dir, sequence_name, image_dir_name, 'data_rect')
        image_arrays = list(map(
            read_image, [os.path.join(image_dir, f"{i:010d}.{self.image_suffix}") for i in img_indexes]
        ))
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]
            data[('original_image', frame_id)] = data[('image', frame_id)].copy()

        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2[0:3, 0:3]
        data['original_P2'] = data['P2'].copy()

        h, w, _ = data[("image", 0)].shape
        if self.read_mask:
            if image_dir_name == 'image_00':
                data['patched_mask'] = self.masks[0].copy()
            if image_dir_name == 'image_01':
                data['patched_mask'] = self.masks[1].copy()
        else:
            data["patched_mask"] = np.ones([h, w])
        data['sequence_name'] = sequence_name
        data['image_index'] = img_indexes[0]

        data = self.transform(deepcopy(data))
        return data

class OdaibaKITTI360Dataset(KITTI360MonoDataset):

    def _load_calib(self):
        cam_calib_file = os.path.join(self.calib_dir, "perspective.txt")
        cam_extrinsic_file = os.path.join(self.calib_dir, "calib_cam_to_pose.txt")

        P0, P1, R0, R1 = read_P01_from_sequence(cam_calib_file)
        T0, T1 = read_extrinsic_from_sequence(cam_extrinsic_file)
        self.cam_calib = dict()
        self.cam_calib['P0'] = P0
        self.cam_calib['T_rect02baselink'] = R0 @ T0

        mask_path = os.path.join(self.raw_path, f"mask.png")
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, -1)
        else:
            self.mask = None

    def __getitem__(self, index):
        obj = self.imdb[index]
        sequence_name = obj['sequence_name']
        pose_indexes = obj['pose_indexes']
        img_indexes = obj['img_indexes']
        extrinsics = self.cam_calib['T_rect02baselink']
        image_dir_name = 'image_00'
        P2 = self.cam_calib['P0']

        
        data = dict()
        poses = self.keypose[sequence_name][pose_indexes] #[3, 4, 4]
        for i, idx in enumerate(self.frame_ids[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(
               poses[0], poses[i+1], np.linalg.inv(extrinsics)
            ).astype(np.float32)
        
        image_dir = os.path.join(self.img_dir, sequence_name, image_dir_name)
        image_arrays = list(map(
            read_image, [os.path.join(image_dir, f"{i:010d}.{self.image_suffix}") for i in img_indexes]
        ))
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]
            data[('original_image', frame_id)] = data[('image', frame_id)].copy()

        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2[0:3, 0:3]
        data['original_P2'] = data['P2'].copy()

        h, w, _ = data[("image", 0)].shape
        if self.mask is None:
            data['patched_mask'] = np.ones([h, w])
        else:
            data['patched_mask'] = self.mask.copy()
        data['sequence_name'] = sequence_name
        data['image_index'] = img_indexes[0]

        data = self.transform(deepcopy(data))
        return data

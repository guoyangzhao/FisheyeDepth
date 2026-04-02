from __future__ import print_function, division
import os
import numpy as np
import cv2
from easydict import EasyDict
import yaml

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader # noqa: F401

import torch
import torch.utils.data
from monodepth.data.datasets.utils import read_image,  cam_relative_pose_nusc, read_csv
from vision_base.utils.builder import build

def read_extrinsic_from_sequence(file):

    T0 = np.eye(4)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T3 = np.eye(4)
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
            if line.startswith("image_02"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T2[0:3, :] = np.reshape(T, [3, 4])
            if line.startswith("image_03"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T3[0:3, :] = np.reshape(T, [3, 4])

    return dict(
        T_image0=T0, T_image1=T1, T_image2=T2, T_image3=T3
    )

def read_fisheycalib(file):
    with open(file, 'r') as f:
        f.readline() #[The first line is not useful and contain not standard yaml]
        calib = yaml.safe_load(f)
    return calib

def extract_P_from_fisheye_calib(calib):
    P = np.zeros([3, 4])
    P[0, 0] = calib["projection_parameters"]["gamma1"]
    P[1, 1] = calib["projection_parameters"]["gamma2"]
    P[0, 2] = calib["projection_parameters"]["u0"]
    P[1, 2] = calib["projection_parameters"]["v0"]
    P[2, 2] = 1
    return P

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


def read_split_file(file:str):
    imdb:list = []
    with open(file, 'r') as f:
        lines = f.readlines() # 2011_09_26/2011_09_26_drive_0022_sync 473 r
        for i in range(len(lines)):
            line = lines[i].strip().split()

            folder = line[0]
            index = int(line[1])
            side = line[2]
            datetime = folder.split("/")[0]
            imdb.append(
                dict(
                    folder=folder,
                    index=index,
                    side=side,
                    datetime=datetime
                )
            )
    return imdb

def read_cam2velo_from_sequence(file):
    """ read T from a sequence file calib_cam_to_velo.txt
    """
    with open(file, 'r') as f:
        line = f.readlines()[0]
        data = line.strip().split(" ")
        T = np.array([float(x) for x in data[0:12]]).reshape([3, 4])

    T_cam2velo = np.eye(4)
    T_cam2velo[0:3, :] = T
    return T_cam2velo

class KITTI360FisheyeDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(KITTI360FisheyeDataset, self).__init__()
        self.raw_path = getattr(data_cfg, 'raw_path', '/data/KITTI-360')
        self.meta_file   = getattr(data_cfg, 'split_file', 'kitti360_meta.txt')
        self.resized_root = getattr(data_cfg, 'resized_root', None)

        if self.resized_root is not None:
            self.img_dir = self.resized_root
            self.calib_dir = os.path.join(self.resized_root, 'calibration')
        else:
            self.img_dir   = os.path.join(self.raw_path, 'data_2d_raw')
            self.calib_dir = os.path.join(self.raw_path, 'calibration')
        self.pose_dir  = os.path.join(self.raw_path, 'data_poses')
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

        fish_eye_mask = getattr(data_cfg, 'fish_eye_mask', None)
        if fish_eye_mask is not None:
            self.fish_eye_mask = cv2.imread(fish_eye_mask, -1)
        else:
            self.fish_eye_mask = None

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
        left_calib_file = os.path.join(self.calib_dir, "image_02.yaml")
        right_calib_file = os.path.join(self.calib_dir, "image_03.yaml")
        cam_extrinsic_file = os.path.join(self.calib_dir, "calib_cam_to_pose.txt")

        left_calib = read_fisheycalib(left_calib_file)
        right_calib = read_fisheycalib(right_calib_file)

        T_image2pose_dict = read_extrinsic_from_sequence(cam_extrinsic_file)

        self.cam_calib = dict()
        self.cam_calib['P0'] = extract_P_from_fisheye_calib(left_calib)
        self.cam_calib['P1'] = extract_P_from_fisheye_calib(right_calib)
        self.cam_calib['T_rect02baselink'] = T_image2pose_dict['T_image2']
        self.cam_calib['T_rect12baselink'] = T_image2pose_dict['T_image3']
        self.cam_calib['left_meta'] = left_calib
        self.cam_calib['right_meta'] = right_calib

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
        # if (not self.use_right_image):
            extrinsics = self.cam_calib['T_rect02baselink']
            image_dir_name = 'image_02'
            P2 = self.cam_calib['P0']
            calib_meta = self.cam_calib['left_meta']
        else:
            extrinsics = self.cam_calib['T_rect12baselink']
            image_dir_name = 'image_03'
            P2 = self.cam_calib['P1']
            calib_meta = self.cam_calib['right_meta']
        
        data = dict()
        poses = self.keypose[sequence_name][pose_indexes] #[3, 4, 4]
        for i, idx in enumerate(self.frame_ids[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(
               poses[0], poses[i+1], np.linalg.inv(extrinsics)
            ).astype(np.float32)
        
        image_dir = os.path.join(self.img_dir, sequence_name, image_dir_name, 'data_rgb')
        image_arrays = list(map(
            read_image, [os.path.join(image_dir, f"{i:010d}.png") for i in img_indexes]
        ))
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]

        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2[0:3, 0:3]
        data['original_P2'] = data['P2'].copy()
        data['calib_meta'] = deepcopy(calib_meta)
        data['model_type'] = calib_meta['model_type']

        h, w, _ = data[("image", 0)].shape
        if self.fish_eye_mask is not None:
            data["patched_mask"] = cv2.resize(self.fish_eye_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            data["patched_mask"] = np.ones([h, w])
        data = self.transform(deepcopy(data))
        return data

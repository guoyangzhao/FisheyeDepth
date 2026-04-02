import numpy as np
import os
import cv2
from PIL import Image
import tqdm
from collections import Counter
from .kitti_unsupervised_eval import KittiEigenEvaluator
from monodepth.networks.utils.monodepth_utils import compute_errors, sub2ind
from monodepth.data.datasets.utils import read_pc_from_bin
from monodepth.data.datasets.utils import read_csv
from monodepth.data.datasets.fisheye_dataset import read_fisheycalib, read_cam2velo_from_sequence
from monodepth.data.datasets.fisheye_dataset import read_extrinsic_from_sequence as read_fisheye_extrinsic, extract_P_from_fisheye_calib
from monodepth.networks.utils.mei_fisheye_utils import MeiCameraProjection


class Kitti360FisheyeEvaluator(KittiEigenEvaluator):
    def __init__(self,
                 data_path, #path to kitti raw data
                 split_file,
                 gt_saved_file,
                 is_evaluate_absolute=False,
                 ):
        self.is_evaluate_absolute = is_evaluate_absolute

        if os.path.isfile(gt_saved_file):
            self.gt_depths = np.load(gt_saved_file, fix_imports=True,encoding='latin1', allow_pickle=True)["data"]
            self.close_masks = np.load(gt_saved_file, fix_imports=True,encoding='latin1', allow_pickle=True)["close_masks"]
        
        else:
            print(f"Start exporting ground truth depths specified by {split_file} to {gt_saved_file}")
            self._precompute(data_path, split_file, gt_saved_file)

    def _load_calib(self, calib_dir):
        left_calib_file = os.path.join(calib_dir, "image_02.yaml")
        right_calib_file = os.path.join(calib_dir, "image_03.yaml")
        cam_extrinsic_file = os.path.join(calib_dir, "calib_cam_to_pose.txt")
        velo_calib_file = os.path.join(calib_dir, "calib_cam_to_velo.txt")

        left_calib = read_fisheycalib(left_calib_file)
        right_calib = read_fisheycalib(right_calib_file)

        T_image2pose_dict = read_fisheye_extrinsic(cam_extrinsic_file)
        T_cam2velo = read_cam2velo_from_sequence(velo_calib_file)

        self.cam_calib = dict()
        self.cam_calib['left_calib'] = left_calib
        self.cam_calib['right_calib'] = right_calib
        self.cam_calib['T_image2pose'] = T_image2pose_dict
        self.cam_calib['P0'] = extract_P_from_fisheye_calib(left_calib)
        self.cam_calib['P1'] = extract_P_from_fisheye_calib(right_calib)
        self.cam_calib['T_cam2velo'] = T_cam2velo

        self.projector = MeiCameraProjection()
    
    def single_call(self, depth_0, index):
        gt_depth = self.gt_depths[index]
        close_mask = self.close_masks[index]
        return self._single_loss(depth_0, gt_depth, close_mask)

    def _single_loss(self, depth_0, gt_depth, close_mask):
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = cv2.resize(depth_0, (gt_width, gt_height))
        
        mask = np.logical_and(gt_depth > 0.3, gt_depth < 60.0)
        mask = np.logical_and(mask, close_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if len(pred_depth) == 0 or len(gt_depth) == 0:
            raise ValueError
        
        ratio = np.median(gt_depth) / np.median(pred_depth)
        scaled_depth = pred_depth * ratio

        scaled_depth[scaled_depth < 1e-3] = 1e-3
        scaled_depth[scaled_depth > 80.0] = 80.0

        error = compute_errors(gt_depth, scaled_depth)

        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80.0] = 80.0

        abs_error = compute_errors(gt_depth, pred_depth)
        return dict(
            ratio = ratio,
            error = error,
            abs_error = abs_error
        )


    def _projection(self, velo_pts_im, norm, im_shape):
        # project to image
        depth = np.zeros((im_shape[:2]))
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

        gt_norm = np.zeros((im_shape[:2]))
        gt_norm[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = norm

        # find the duplicate points and choose the closest depth
        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1] # type: ignore
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
            gt_norm[y_loc, x_loc] = norm[pts].min()
        depth[depth < 0] = 0
        gt_norm[gt_norm < 0] = 0

        return depth, gt_norm

    def _precompute(self, data_path, split_file, gt_saved_file):
        img_dir   = os.path.join(data_path, 'data_2d_raw')
        calib_dir = os.path.join(data_path, 'calibration')
        pc_dir    = os.path.join(data_path, 'data_3d_raw')
        self._load_calib(calib_dir)

        T_cam002pose = self.cam_calib['T_image2pose']['T_image0']
        T_cam022pose = self.cam_calib['T_image2pose']['T_image2']
        T_velo2cam02 = T_velo2cam02 = np.linalg.inv(T_cam022pose) @ T_cam002pose @ np.linalg.inv(self.cam_calib['T_cam2velo'])

        gt_depths = []
        masks = []

        header, data = read_csv(split_file)
        for data_cell in tqdm.tqdm(data, dynamic_ncols=True):
            sequence_name = data_cell['sequence_name']
            frame_id = int(data_cell['0'])

            velo_filename = os.path.join(pc_dir, sequence_name,"velodyne_points/data", "{:010d}.bin".format(frame_id))
            velo = read_pc_from_bin(velo_filename)
            velo_camera_frame = (T_velo2cam02 @ np.concatenate(
                [velo[:, 0:3], np.ones([velo.shape[0], 1])], axis=1).T).T[:, 0:3]  #[N, 4]
            
            velo_camera_frame = velo_camera_frame[velo_camera_frame[:, 2]  > 0]

            image_filename = os.path.join(img_dir, sequence_name, 'image_02', 'data_rgb', "{:010d}.png".format(frame_id))
            pil_image = Image.open(image_filename)
            image_shape = np.array(pil_image.size)[::-1].astype(np.int32) #[h, w]

            velo_pts_im = self.projector.cam2image(velo_camera_frame,
                                                   self.cam_calib['P0'],
                                                   self.cam_calib['left_calib']) #[x, y, norm]
            velo_pts_im[:, 2] = velo_camera_frame[:, 2]
            norm = np.linalg.norm(velo_camera_frame[:, 0:3], axis=1)
            # velo_pts_im[:, 2] = norm
            gt_depth, gt_norm = self._projection(velo_pts_im, norm, image_shape)
            mask = (gt_norm > 0) * (gt_norm < 8)

            gt_depths.append(gt_depth.astype(np.float32))
            masks.append(mask.astype(bool))
        
        np.savez_compressed(gt_saved_file, data=np.array(gt_depths), close_masks=masks)
        self.gt_depths = gt_depths
        self.close_masks = masks

import numpy as np
import os
import cv2
from PIL import Image
import tqdm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from monodepth.networks.utils.monodepth_utils import generate_depth_map, compute_errors, project_depth_map
from monodepth.data.datasets.utils import read_depth, read_pc_from_bin, read_csv
from monodepth.data.datasets.kitti360_dataset import read_extrinsic_from_sequence, read_P01_from_sequence, read_T_from_sequence

class KittiEigenEvaluator(object):
    def __init__(self,
                 data_path, #path to kitti raw data
                 split_file,
                 gt_saved_file,
                 is_evaluate_absolute=False,
                 ):
        self.is_evaluate_absolute = is_evaluate_absolute

        if os.path.isfile(gt_saved_file):
            self.gt_depths = np.load(gt_saved_file, fix_imports=True,encoding='latin1', allow_pickle=True)["data"]
        
        else:
            print(f"Start exporting ground truth depths specified by {split_file} to {gt_saved_file}")
            self._precompute(data_path, split_file, gt_saved_file)

    def _precompute(self, data_path, split_file, gt_saved_file):
        with open(split_file, 'r' ) as f:
            lines = f.readlines()
        
        gt_depths = []
        for line in lines:
            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)

            calib_dir = os.path.join(data_path, folder.split("/")[0])
            velo_filename = os.path.join(data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
            gt_depths.append(gt_depth.astype(np.float32))
        # np.savez_compressed(gt_saved_file, data=np.array(gt_depths))
        self.gt_depths = gt_depths


    def _single_loss(self, depth_0, gt_depth):
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = cv2.resize(depth_0, (gt_width, gt_height))
        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

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

    def single_call(self, depth_0, index):
        gt_depth = self.gt_depths[index]
        return self._single_loss(depth_0, gt_depth)

    def log(self, writer, mean_errors, mean_abs_errors, global_step=0, epoch_num=0, is_print=True):
        log_str = f"Epoch {epoch_num}"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"

        log_str += f"\nEpoch {epoch_num}| Abs Error without Scaled"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_abs_errors.tolist()) + "\\\\"

        if writer is not None:
            writer.add_text(f"evaluation logs", log_str.replace(' ', '&nbsp;').replace('\n', '  \n'), global_step=epoch_num)
        
        if is_print:
            print(log_str)
    

    def __call__(self, result_path, writer:Optional[SummaryWriter]=None, global_step=0, epoch_num=0):
        filelist = os.listdir(result_path)
        filelist.sort()

        if len(filelist) != len(self.gt_depths):
            print(f"The length of pred_depths is {len(filelist)} while the length of gt_depths is {len(self.gt_depths)}")
            print(f"Drop evaluation")
            return

        errors = []
        abs_errors = []
        ratios = []
        for i, image_file in enumerate(tqdm.tqdm(filelist, dynamic_ncols=True)):

            gt_depth = self.gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = read_depth(os.path.join(result_path, image_file))
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

            mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            scaled_depth = pred_depth * ratio

            scaled_depth[scaled_depth < 1e-3] = 1e-3
            scaled_depth[scaled_depth > 80.0] = 80.0

            errors.append(compute_errors(gt_depth, scaled_depth))

            pred_depth[pred_depth < 1e-3] = 1e-3
            pred_depth[pred_depth > 80.0] = 80.0

            abs_errors.append(compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(errors).mean(0)
        mean_abs_errors = np.array(abs_errors).mean(0)
        scales = np.array(ratios)
        
        log_str = f"Epoch {epoch_num} | Scaled Error | {scales.mean()}, {scales.std()}"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"

        log_str += f"\nEpoch {epoch_num} | Abs Error without Scaled"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_abs_errors.tolist()) + "\\\\"

        if writer is not None:
            writer.add_text("evaluation logs", log_str.replace(' ', '&nbsp;').replace('\n', '  \n'), global_step=epoch_num)
        print(log_str)

    
class Kitti360Evaluator(KittiEigenEvaluator):

    def _load_calib(self, calib_dir):
        cam_calib_file = os.path.join(calib_dir, "perspective.txt")
        cam_extrinsic_file = os.path.join(calib_dir, "calib_cam_to_pose.txt")
        velo_calib_file = os.path.join(calib_dir, "calib_cam_to_velo.txt")

        T_cam2velo = read_T_from_sequence(velo_calib_file)
        P0, P1, R0, R1 = read_P01_from_sequence(cam_calib_file)
        T0, T1 = read_extrinsic_from_sequence(cam_extrinsic_file)
        self.cam_calib = dict()
        self.cam_calib['P0'] = P0
        self.cam_calib['R0'] = R0
        self.cam_calib['T_cam2velo'] = T_cam2velo


    def _precompute(self, data_path, split_file, gt_saved_file):
        img_dir   = os.path.join(data_path, 'data_2d_raw')
        calib_dir = os.path.join(data_path, 'calibration')
        pc_dir    = os.path.join(data_path, 'data_3d_raw')

        self._load_calib(calib_dir)

        header, data = read_csv(split_file)
        
        P_velo2img = self.cam_calib['P0'] @ self.cam_calib['R0'] @ np.linalg.inv(self.cam_calib['T_cam2velo'])
        gt_depths = []
        for data_cell in tqdm.tqdm(data, dynamic_ncols=True):
            sequence_name = data_cell['sequence_name']
            frame_id = int(data_cell['0'])

            velo_filename = os.path.join(pc_dir, sequence_name,"velodyne_points/data", "{:010d}.bin".format(frame_id))
            velo = read_pc_from_bin(velo_filename)

            image_filename = os.path.join(img_dir, sequence_name, 'image_00', 'data_rect', "{:010d}.png".format(frame_id))
            pil_image = Image.open(image_filename)
            image_shape = np.array(pil_image.size)[::-1].astype(np.int32) #[h, w]

            gt_depth = project_depth_map(velo, P_velo2img, image_shape)

            gt_depths.append(gt_depth.astype(np.float32))
        
        np.savez_compressed(gt_saved_file, data=np.array(gt_depths))
        self.gt_depths = gt_depths

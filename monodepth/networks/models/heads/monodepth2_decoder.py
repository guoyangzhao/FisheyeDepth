# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_base.utils.builder import build

from monodepth.networks.utils.monodepth_utils import BackprojectDepth, Project3D, SSIM, compute_depth_errors, get_smooth_loss
from monodepth.networks.utils.mei_fisheye_utils import MeiCameraProjection
from monodepth.networks.utils.opencv_fisheye_utils import OpenCVFisheyeCameraProjection

class MonoDepth2Decoder(nn.Module):
    def __init__(self,
                 scales,
                 height, width,
                 frame_ids,
                 depth_decoder_cfg,
                 pose_decoder_cfg=None,
                 multiscale_head_cfg=None,
                 **kwargs
                ):
        super(MonoDepth2Decoder, self).__init__()
        self.scales=scales
        self.num_scales = len(scales)
        self.height=height
        self.width = width
        self.frame_ids = frame_ids
        self.depth_decoder = build(**depth_decoder_cfg)
        if pose_decoder_cfg is not None:
            self.pose_decoder = build(**pose_decoder_cfg)
        if multiscale_head_cfg is not None:
            self.multiscale_head = build(**multiscale_head_cfg)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.backproject_depth = BackprojectDepth()
        self.project_3d = Project3D()
        self.ssim = SSIM()

        for key in kwargs:
            setattr(self, key, kwargs[key])


    def forward_pose(self, *args, **kwargs):
        axisangle, translation = self.pose_decoder(*args, **kwargs)

        return axisangle, translation

    def forward_depth(self, features, *args, **kwargs):
        outputs  = self.depth_decoder(features, *args, **kwargs)
        return outputs

    def _generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            
            depth_before = outputs[('depth', scale, scale)]
            depth = F.interpolate(
                depth_before, [self.height, self.width], mode="bilinear", align_corners=True)

            batch, _, height, width = depth.shape
            
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id)]

                K = np.zeros([batch, 4, 4])
                K[:, 0:3, 0:3] = inputs['P2'][:, 0:3, 0:3].cpu().numpy()
                K[:, 3, 3] = 1
                inv_K = np.linalg.pinv(K)

                cam_points = self.backproject_depth(
                    depth, torch.from_numpy(inv_K).cuda().float()) # all camera points in frame 0
                pix_coords = self.project_3d.forward(
                    cam_points, torch.from_numpy(K).cuda().float(), T, batch, height, width) # camera points in frame 0, presented in frame 1

                is_residual_flow = getattr(self, 'is_residual_flow', False)
                if is_residual_flow and ('residual_flow', frame_id, 0) in outputs:
                    residual_flow = outputs[('residual_flow', frame_id, 0)] # [B, 2, H, W]
                    residual_flow = residual_flow.permute(0, 2, 3, 1).contiguous() #[B, H, W, 2]
                    pix_coords = pix_coords + residual_flow

                outputs[("original_image", frame_id, scale)] = F.grid_sample(
                    inputs[("original_image", frame_id)],
                    pix_coords,
                    padding_mode="border", align_corners=True)

                is_light_compensate = getattr(self, 'is_light_compensate', False)
                if is_light_compensate and ('light_compensate_ct', frame_id, 0) in outputs:
                    C_t = outputs[('light_compensate_ct', frame_id, 0)] #[B, 1, H, W]
                    B_t = outputs[('light_compensate_bt', frame_id, 0)] #[B, 1, H, W]
                    outputs[('original_image', frame_id, scale)] = \
                        outputs[('original_image', frame_id, scale)] * (1 + C_t) + B_t

                overlapped_mask_on = getattr(self,'overlapped_mask', False)
                if overlapped_mask_on:
                    key = ("patched_mask", frame_id) if ("patched_mask", frame_id) in inputs else "patched_mask"
                    patched_mask = inputs.get(key, torch.ones([batch, height, width]).cuda()) #[B, H, W]
                    reproject_patch = F.grid_sample(
                        patched_mask.unsqueeze(1).float(),
                        pix_coords, align_corners=True, mode='nearest') #[B, 1, H, W]
                    outputs[("overlapped_mask", frame_id, scale)] = (reproject_patch == 1).squeeze(1) #[B, H, W]

    def compute_reprojection_loss(self, pred, target, ssim_weight=0.85):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

  
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = ssim_weight * ssim_loss + (1 - ssim_weight) * l1_loss

        return reprojection_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def get_prediction(self, input_dict, output_dict):
        return dict(depth= output_dict[("depth", 0, 0)])

    def compute_similarity_weight(self, input_dict, output_dict):
        ssim_layer = SSIM(kernel_size=13, padding=6)
        image0 = input_dict[('original_image', 0)]

        ssims = []
        for frame_id in self.frame_ids[1:]:
            image1 = input_dict[('original_image', frame_id)]
            ssim = ssim_layer(image0, image1) #[B, 3, H, W]
            ssims.append(ssim)
        ssim_linked = torch.cat(ssims, dim=1).mean(dim=1) #[B, H, W]
        weights = ssim_linked / ssim_linked.mean(dim=[1, 2], keepdim=True) #[B, H, W]
        return weights

    def compute_pose_loss(self, output_dict, input_dict):
        pose_loss = 0
        for frame_id in self.frame_ids[1:]:
            target_T = input_dict[('relative_pose', frame_id)]
            predict_T = output_dict[("cam_T_cam", frame_id)]
            pose_loss = pose_loss + torch.abs(target_T - predict_T).mean()

        return pose_loss
    
    def compute_distill_loss(self, output_dict, input_dict, scale):
        prediction_depth = output_dict[('depth', scale, scale)]
        teacher_depth    = output_dict[('teacher_depth', scale, scale)].detach()

        is_unscaled_distill = getattr(self, 'is_unscaled_distill', False)
        if is_unscaled_distill:
            ratio_diff = (prediction_depth / (teacher_depth + 1e-5)
                ).mean(dim=[2, 3], keepdim=True)
            error = (ratio_diff * teacher_depth - prediction_depth).abs()
        else:
            error = (teacher_depth - prediction_depth).abs()

        is_uncertain_distill = getattr(self, 'is_uncertain_distill', False)
        if is_uncertain_distill:
            uncertain_logz   = output_dict[('uncertain_z', scale)]
            loss  = error / uncertain_logz + torch.log(uncertain_logz + 1e-5)
        else:
            loss = error
        return loss.mean()

    def compute_total_reprojection_loss(self, output_dict, input_dict):
        self._generate_images_pred(input_dict, output_dict)
        losses = {}
        hm = {}
        total_loss = 0
        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            disp = output_dict[("disp", scale)]
            if scale == 0:
                color = input_dict[("original_image", 0)]
            else:
                _, _, h, w = disp.shape
                color = F.adaptive_avg_pool2d(input_dict[("original_image", 0)], (h, w))
             
            target = input_dict[("original_image", 0)]
            B, C, H, W = target.shape
            hm[f'original_image'] = target[0:1]
            

            for frame_id in self.frame_ids[1:]:
                pred = output_dict[("original_image", frame_id, scale)]
                projection_loss = self.compute_reprojection_loss(pred, target)

                # mask out non_overlapped_mask
                overlapped_mask_on = getattr(self,'overlapped_mask', False)
                if overlapped_mask_on:
                    overlapped_mask = output_dict[("overlapped_mask", frame_id, scale)].unsqueeze(1)
                    non_overlapped_mask = torch.logical_not(overlapped_mask)
                    projection_loss[non_overlapped_mask] = 100.0 # a large number for each pixel; block gradients and be omitted by min
                reprojection_losses.append(projection_loss)
                if scale == 0:
                    hm[f'predicted_image_{frame_id}'] = pred[0:1]

            reprojection_losses = torch.cat(reprojection_losses, 1)

            
            if 'motion_mask' in input_dict:
                motion_mask = input_dict['motion_mask']
                to_optimise, idxs = torch.min(reprojection_losses, dim=1)
                to_optimise = to_optimise.detach() * motion_mask + to_optimise * (1 - motion_mask)
            else:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = input_dict[("original_image", frame_id)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)


                # add random numbers to break ties
                identity_reprojection_losses += torch.randn(
                    identity_reprojection_losses.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)

                to_optimise, idxs = torch.min(combined, dim=1)
                
                if scale==0:
                    hm[f'loss_mask_{scale}'] = dict(
                        data=(idxs >= 2).unsqueeze(1)[0:1]
                    )
            
            key = ("patched_mask", 0) if ("patched_mask", 0) in input_dict else "patched_mask"
            patched_mask = input_dict.get(key, torch.ones([B, H, W]).cuda()) #[B, H, W] #[B, 1, H, W]
            ssim_weight  = output_dict.get("ssim_weight", torch.ones([B, H, W]).cuda())
            to_optimise = to_optimise * patched_mask * ssim_weight
            #to_optimise = torch.mean(reprojection_losses, dim=1)

            #output_dict["identity_selection/{}".format(scale)] = (
            #    idxs > identity_reprojection_losses.shape[1] - 1).float()

            learnable_photometric_uncertain = getattr(self,'learnable_photometric_uncertain', False)
            if learnable_photometric_uncertain:
                photometric_uncertain = self.photometric_net(
                    torch.cat(
                        [input_dict[("original_image", frame_id)] for frame_id in self.frame_ids] +
                        [output_dict[("original_image", frame_id, scale)] for frame_id in self.frame_ids[1:]],
                        dim = 1
                    ) #[B, 5 * 3, H, W]
                )
                photometric_net_grad_weight = getattr(self, 'photometric_net_grad_weight', 0.05)
                photometric_uncertain = photometric_net_grad_weight * photometric_uncertain + \
                        (1 - photometric_net_grad_weight) * photometric_uncertain.detach()
                to_optimise = to_optimise / photometric_uncertain + torch.log(photometric_uncertain + 1e-5)

            loss += to_optimise.sum() / (patched_mask.sum() + 1e-6)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color) * 1e-5 / (2 ** scale)
            losses[f'smooth_loss/{scale}'] = smooth_loss.detach()

            loss += smooth_loss
            total_loss += loss
            losses["loss/{}".format(scale)] = loss.detach()

        total_loss /= self.num_scales
        return losses, hm, total_loss

    def loss(self, output_dict, input_dict):
        losses = {}
        hm = {}
        total_loss = 0

        is_ssim_weight = getattr(self, 'is_ssim_weight', False)
        if is_ssim_weight:
            ssim_weight = self.compute_similarity_weight(input_dict)
            output_dict['ssim_weight'] = ssim_weight

        repro_loss_dict, repro_loss_hm, repro_total_loss = self.compute_total_reprojection_loss(output_dict, input_dict)
        losses.update(repro_loss_dict)
        hm.update(repro_loss_hm)
        total_loss += repro_total_loss

        # Learn Pose
        pose_weight = getattr(self, 'pose_loss_weight', 0)
        if pose_weight > 0:
            pose_loss = self.compute_pose_loss(output_dict, input_dict)
            losses['pose_loss'] = pose_loss
            total_loss = total_loss + pose_weight * pose_loss

        # Distillation
        distillation_weight = getattr(self, 'distillation_loss_weight', 0)
        if distillation_weight > 0:
            for scale in self.scales:
                distilation_loss = self.compute_distill_loss(output_dict, input_dict, scale)
                losses["distilation/{}".format(scale)] = distilation_loss.detach()
                total_loss = total_loss + distilation_loss * distillation_weight

        residualflow_weight = getattr(self, 'residualflow_weight', 0)
        if residualflow_weight > 0:
            for scale in self.multiscale_head.scales:
                residualflow_loss = self.compute_residualflow_loss(output_dict, input_dict, scale)
                losses["residualflow_weight/{}".format(scale)] = residualflow_loss.detach()
                total_loss = total_loss + residualflow_loss * residualflow_weight

        losses["total_loss"] = total_loss.detach()
        is_log_image = getattr(self, 'is_log_image', True)
        if not is_log_image:
            hm = {}
        return {'loss':total_loss, 'loss_dict': losses, 'hm':hm}


class FishEyeDecoder(MonoDepth2Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_dict = {
            'PinHole': OpenCVFisheyeCameraProjection(),
            'MEI': MeiCameraProjection(),
        }

    def _generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            depth_before = outputs[('depth', scale, scale)]
            depth = F.interpolate(
                depth_before, [self.height, self.width], mode="bilinear", align_corners=True)

            batch, _, height, width = depth.shape
            
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id)]
                    # T = torch.eye(4)[None].cuda()

                P = inputs["P2"]
                calib_meta = inputs["calib_meta"]
                points, mask = [], []
                for b in range(batch):
                    projector = self.project_dict[inputs['model_type'][b]]
                    pts, msk = projector.image2cam(depth[b:b+1], P[b:b+1], [calib_meta[b]])
                    points.append(pts)
                    mask.append(msk)
                points = torch.cat(points, dim=0)
                mask = torch.cat(mask, dim=0)
                homo_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).squeeze(1)[..., None] #[B, H, W, 4, 1]

                T_reshaped = T[:, None, None] #[2, 1, 1, 4, 4]
                transformed_points = torch.matmul(T_reshaped, homo_points)[..., 0] # [B, H, W, 4]
                pix_coords = torch.stack([
                    self.project_dict[inputs['model_type'][b]].cam2image(transformed_points[b, ..., 0:3], P[b], calib_meta[b]) for b in range(batch)
                ], dim=0)[..., 0:2] #[b, H, W, 2]
                normalized_pix = torch.stack(
                    [pix_coords[..., 0] / max(width - 1, 1) * 2 - 1, pix_coords[..., 1] / max(height - 1, 1) * 2 - 1], dim=-1
                )  # [B, H, W, 2]

                is_residual_flow = getattr(self, 'is_residual_flow', False)
                if is_residual_flow and ('residual_flow', frame_id, 0) in outputs:
                    residual_flow = outputs[('residual_flow', frame_id, 0)] # [B, 2, H, W]
                    residual_flow = residual_flow.permute(0, 2, 3, 1).contiguous() #[B, H, W, 2]
                    normalized_pix = normalized_pix + residual_flow

                outputs[("original_image", frame_id, scale)] = F.grid_sample(
                    inputs[("original_image", frame_id)],
                    normalized_pix,
                    padding_mode="border", align_corners=True)

                is_light_compensate = getattr(self, 'is_light_compensate', False)
                if is_light_compensate and ('light_compensate_ct', frame_id, 0) in outputs:
                    C_t = outputs[('light_compensate_ct', frame_id, 0)] #[B, 1, H, W]
                    B_t = outputs[('light_compensate_bt', frame_id, 0)] #[B, 1, H, W]
                    outputs[('original_image', frame_id, scale)] = \
                        outputs[('original_image', frame_id, scale)] * (1 + C_t) + B_t

                overlapped_mask_on = getattr(self,'overlapped_mask', False)
                if overlapped_mask_on:
                    Z_mask = transformed_points[..., 2] > 0  #[B, H, W]
                    patched_mask = inputs.get("patched_mask", torch.ones([batch, height, width]).cuda())  #[B, H, W]
                    reproject_patch = F.grid_sample(
                        patched_mask.unsqueeze(1).float(),
                        normalized_pix, align_corners=True, mode='nearest').detach() * Z_mask.unsqueeze(1) * mask #[B, 1, H, W]
                    outputs[("overlapped_mask", frame_id, scale)] = (reproject_patch == 1).squeeze(1) #[B, H, W]
    
    def get_prediction(self, input_dict, output_dict):
        norm = output_dict[("depth", 0, 0)]
        P = input_dict["P2"]
        calib_meta = input_dict["calib_meta"][0]
        points, mask = self.project_dict[input_dict['model_type'][0]].image2cam(norm, P, [calib_meta])
        return dict(depth= points[..., 2], norm=norm)

import torch
from easydict import EasyDict
from typing import Optional
from vision_base.utils.builder import build
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
from monodepth.networks.utils.monodepth_utils import transformation_from_parameters

class MonoDepthMeta(BaseMetaArch):
    """Some Information about MonoDepthMeta"""
    def __init__(self, depth_backbone_cfg:EasyDict,
                       pose_backbone_cfg:EasyDict,
                       head_cfg:EasyDict,
                       train_cfg:EasyDict,
                       test_cfg:EasyDict,
                       **kwargs,
                       ):
        super(MonoDepthMeta, self).__init__()
        self.depth_backbone = build(**depth_backbone_cfg)
        self.pose_backbone  = build(**pose_backbone_cfg)
        self.head           = build(frame_ids=train_cfg.frame_ids, **head_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def forward_train(self, data, meta):
        image_0 = data[('image', 0)]
        features = self.depth_backbone(image_0)
        outputs = self.head.forward_depth(features)

        for f_i in self.train_cfg.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [data[('image', f_i)], data[('image', 0)]]
            else:
                pose_inputs = [data[('image', 0)], data[('image', f_i)]]

            pose_inputs = [self.pose_backbone(torch.cat(pose_inputs, 1))]
            axisangle, translation = self.head.forward_pose(pose_inputs)

            outputs[("axisangle", f_i)] = axisangle
            outputs[("translation", f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        
        return_dict = self.head.loss(outputs, data)
        return return_dict

    def dummy_forward(self, image):
        features = self.depth_backbone(image)
        outputs = self.head.forward_depth(features)
        depth_prediction = self.head.get_prediction(None, outputs)
        return depth_prediction

    def forward_test(self, data, meta):
        features = self.depth_backbone(data[('image', 0)])
        outputs = self.head.forward_depth(features)
        depth_prediction = self.head.get_prediction(data, outputs)
        return depth_prediction

    def forward(self, data, meta):
        if meta['is_training']:
            return self.forward_train(data, meta)
        else:
            return self.forward_test(data, meta)

class MonoDepthWPose(BaseMetaArch):
    def __init__(self, depth_backbone_cfg:EasyDict,
                       head_cfg:EasyDict,
                       train_cfg:EasyDict,
                       test_cfg:EasyDict,
                       pose_backbone_cfg:Optional[EasyDict]=None,
                       **kwargs,
                       ):
        super(MonoDepthWPose, self).__init__()
        self.depth_backbone = build(**depth_backbone_cfg)
        self.head           = build(frame_ids=train_cfg.frame_ids, **head_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.is_use_res_pose = pose_backbone_cfg is not None
        if pose_backbone_cfg is not None:
            self.pose_backbone = build(**pose_backbone_cfg)

    
    def forward_train(self, data, meta):
        depth_production_frames = getattr(self.train_cfg, 'depth_production_frames', [0])
        outputs = {}
        for f_i in depth_production_frames:
            image_0 = data[('image', 0)]
            features = self.depth_backbone(image_0)
            output_f_i = self.head.forward_depth(features, data['P2'])
            if f_i == 0:
                outputs.update(output_f_i)
            else:
                for key in output_f_i:
                    if key[0] == 'depth':
                        new_key = (f"depth_{f_i}", key[1], key[2])
                        outputs[new_key] = outputs[key]

        if self.is_use_res_pose:
            for f_i in self.train_cfg.frame_ids[1:]:
                if f_i < 0:
                    pose_inputs = [data[('image', f_i)], data[('image', 0)]]
                    base_pose = data[('relative_pose', f_i)]
                else:
                    pose_inputs = [data[('image', 0)], data[('image', f_i)]]
                    base_pose = torch.linalg.inv(data[('relative_pose', f_i)])

                pose_inputs = [self.pose_backbone(torch.cat(pose_inputs, 1))]
                axisangle, translation = self.head.forward_pose(pose_inputs, base_pose)

                outputs[("axisangle", f_i)] = axisangle
                outputs[("translation", f_i)] = translation

                # Invert the matrix if the frame id is negative
                T = torch.matmul(
                    data[('relative_pose', f_i)], transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                ) #[B, 4, 4]
                ratio = torch.norm(T[:, :3, 3]) / torch.norm(data[('relative_pose', f_i)][:, :3, 3]) #[B, ]
                scale = torch.ones(T.shape[0], 4, 4).cuda()
                scale[:, :3, 3] = ratio
                outputs[("cam_T_cam", f_i)] = T / scale # [B, 4, 4]  * [B, :, 4]
                
        else:
            for f_i in self.train_cfg.frame_ids[1:]:
                outputs[("cam_T_cam", f_i)] = data[('relative_pose', f_i)]
        
        return_dict = self.head.loss(outputs, data)
        return return_dict

    def forward_test(self, data, meta):
        features = self.depth_backbone(data[('image', 0)])
        outputs = self.head.forward_depth(features, data['P2'])
        depth_prediction = self.head.get_prediction(data, outputs)
        return depth_prediction
    
    def dummy_forward(self, image, P):
        features = self.depth_backbone(image)
        outputs = self.head.forward_depth(features, P)
        depth_prediction = self.head.get_prediction(None, outputs)
        return depth_prediction

    def forward(self, data, meta):
        if meta['is_training']:
            return self.forward_train(data, meta)
        else:
            return self.forward_test(data, meta)

class DistillWPoseMeta(BaseMetaArch):
    def __init__(self, teacher_net_cfg:EasyDict,
                       depth_backbone_cfg:EasyDict,
                       # photo_uncer_cfg:EasyDict,
                       teacher_net_path:str,
                       head_cfg:EasyDict,
                       train_cfg:EasyDict,
                       test_cfg:EasyDict,
                       **kwargs):
        super(DistillWPoseMeta, self).__init__()
        self.teacher_net = build(**teacher_net_cfg)
        self.teacher_net.load_state_dict(
            torch.load(teacher_net_path, map_location='cpu'),
            strict=False
        )
        for param in self.teacher_net.parameters():
            param.requires_grad=False

        self.depth_backbone = build(**depth_backbone_cfg)
        self.head           = build(frame_ids=train_cfg.frame_ids, **head_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def train(self, mode=True):
        super(DistillWPoseMeta, self).train(mode)
        self.teacher_net.eval()

    def forward_train(self, data, meta):
        image_0 = data[('image', 0)]
        features = self.depth_backbone(image_0)
        outputs:dict = self.head.forward_depth(features, data['P2'])
        teacher_output:dict = self.teacher_net.compute_teacher_depth(image_0)
        outputs.update(teacher_output)
        
        for f_i in self.train_cfg.frame_ids[1:]:
            outputs[("cam_T_cam", f_i)] = data[('relative_pose', f_i)]

        return_dict = self.head.loss(outputs, data)
        return return_dict

    def forward_test(self, data, meta):
        features = self.depth_backbone(data[('image', 0)])
        outputs = self.head.forward_depth(features, data['P2'])
        depth_prediction = self.head.get_prediction(data, outputs)
        return depth_prediction

    def forward(self, data, meta):
        if meta['is_training']:
            return self.forward_train(data, meta)
        else:
            return self.forward_test(data, meta)

    def dummy_forward(self, image):
        features = self.depth_backbone(image)
        outputs = self.head.forward_depth(features)
        depth_prediction = self.head.get_prediction(None, outputs)
        return depth_prediction

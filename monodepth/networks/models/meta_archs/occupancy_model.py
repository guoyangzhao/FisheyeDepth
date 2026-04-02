import torch
from easydict import EasyDict
from vision_base.utils.builder import build
from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch


class OccupancyModel(BaseMetaArch):
    def __init__(self, backbone_cfg:EasyDict,
                       teacher_net_cfg:EasyDict,
                       teacher_net_path:str,
                       head_cfg:EasyDict,
                       train_cfg:EasyDict,
                       test_cfg:EasyDict,
                       **kwargs,
                       ):
        super(OccupancyModel, self).__init__()
        self.backbone  = build(**backbone_cfg)
        self.teacher_net = build(**teacher_net_cfg)
        self.teacher_net.load_state_dict(
            torch.load(teacher_net_path, map_location='cpu'),
            strict=False
        )
        for param in self.teacher_net.parameters():
            param.requires_grad=False

        self.head      = build(frame_ids=train_cfg.frame_ids, **head_cfg)
        self.train_cfg = train_cfg
        self.test_cfg  = test_cfg

    def train(self, mode=True):
        super(OccupancyModel, self).train(mode)
        self.teacher_net.eval()
    
    def forward_train(self, data, meta):
        image_0 = data[('image', 0)]
        features = self.backbone(image_0)
        outputs = self.head(features, data['P2'], data)
        with torch.no_grad():
            for idx in self.train_cfg.frame_ids:
                outputs[('teacher_depth', idx, 0)] = self.teacher_net.compute_teacher_depth(data[('image', idx)])[('teacher_depth', 0, 0)]
        return_dict = self.head.loss(outputs, data)
        return return_dict

    def forward_test(self, data, meta):
        features = self.backbone(data[('image', 0)])
        outputs = self.head(features, data['P2'], data)

        with torch.no_grad():
            outputs[('teacher_depth', 0, 0)] = self.teacher_net.compute_teacher_depth(data[('image', 0)])[('teacher_depth', 0, 0)]
        output = dict()
        output['student'] = self.head.get_prediction(data, outputs)
        # output['teacher'] = self.head.get_teacher_voxel(outputs[('teacher_depth', 0, 0)], outputs, data['P2'])
        return output
    

    def forward(self, data, meta):
        if meta['is_training']:
            return self.forward_train(data, meta)
        else:
            return self.forward_test(data, meta)

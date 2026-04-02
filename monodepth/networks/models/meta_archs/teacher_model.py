import torch
import torch.nn as nn
from vision_base.utils.builder import build

class MonoDepthInference(nn.Module):
    def __init__(self, backbone_cfg,
                       depth_head_cfg,
                       is_produce_detached=True,
                       **kwargs):
        super(MonoDepthInference, self).__init__()
        self.depth_backbone = build(**backbone_cfg)
        self.depth_decoder  = build(**depth_head_cfg)
        self.is_produce_detached = is_produce_detached

    def forward(self, x):
        features = self.depth_backbone(x)
        output_dict = self.depth_decoder(features)
        return output_dict

    def compute_teacher_depth(self, x):
        if self.is_produce_detached:
            with torch.no_grad():
                output_dict = self(x)
        else:
            output_dict = self(x)
        teacher_output = {}
        for key in output_dict:
            if key[0] == 'depth':
                new_key = ("teacher_depth", key[1], key[2])
                teacher_output[new_key] = output_dict[key]
        
        return teacher_output

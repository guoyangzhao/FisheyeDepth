# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from vision_base.networks.blocks.blocks import ConvBnReLU
from monodepth.networks.utils.monodepth_utils import depth_to_disp, disp_to_depth

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, min_depth=0.1, max_depth=100, base_fx=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.base_fx = base_fx

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._build_depth_bins(min_depth, max_depth, num_output_channels)

        self._init_layers()

    def _get_scale(self, P2):
        if (self.base_fx is None) or (P2 is None):
            depth_scale = 1
        else:
            input_fx = P2[:, 0, 0] #[B]
            depth_scale = input_fx / self.base_fx #[B]
            depth_scale = depth_scale.reshape([-1, 1, 1, 1]) #[B, 1, 1, 1]
        return depth_scale
    
    def _init_layers(self):
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBnReLU(num_ch_in, num_ch_out, kernel_size=(3, 3))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBnReLU(num_ch_in, num_ch_out, kernel_size=(3, 3), padding_mode='replicate')

        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], self.num_output_channels, kernel_size=3, padding=1, padding_mode='replicate')
            #nn.init.constant_(self.convs[('dispconv', s)].bias, -6)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def _build_depth_bins(self, min_depth, max_depth, num_bins):
        inv_depth_min = np.log(min_depth)
        inv_depth_max = np.log(max_depth)

        inv_depth_bins = torch.arange(inv_depth_min, inv_depth_max, (inv_depth_max - inv_depth_min) / num_bins)
        depth_bins = torch.exp(inv_depth_bins)
        self.register_buffer("depth_bins", depth_bins)

    def _gather_activation(self, x:torch.Tensor)->torch.Tensor:
        """Decode the output of the cost volume into a encoded depth feature map.

        Args:
            x (torch.Tensor): The output of the cost volume of shape [B, num_depth_bins, H, W]

        Returns:
            torch.Tensor: Encoded depth feature map of shape [B, 1, H, W]
        """
        x = torch.clamp(x, -10.0, 10.0)
        activated = torch.softmax(x, dim=1)
        encoded_depth = torch.sum(activated * self.depth_bins.reshape(1, -1, 1, 1), dim=1, keepdim=True) # type: ignore
        return encoded_depth

    def forward(self, input_features, P2=None):
        outputs = {}

        depth_scale = self._get_scale(P2)
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs_logits = self.convs[("dispconv", i)](x)
                outputs[('logits', i)] = outputs_logits
                outputs[("disp", i)] = self.sigmoid(outputs_logits)
                #outputs[("disp", i)] = self.sigmoid(self._gather_activation(self.convs[("dispconv", i)](x)))
                _, depth = disp_to_depth(outputs[("disp", i)], self.min_depth * depth_scale, self.max_depth * depth_scale)
                outputs[("depth", i, i)] = depth

        return outputs

    
class MultiChannelDepthDecoder(DepthDecoder):
    def gather_output(self, output_logits, depth_scale):
        if self.base_fx is not None:
            depth = self._gather_activation(output_logits) * depth_scale
        else:
            depth = self._gather_activation(output_logits)
        disp = depth_to_disp(depth, self.min_depth * depth_scale, self.max_depth * depth_scale)
        return depth, disp

    def forward(self, input_features, P2=None):
        outputs = {}
        depth_scale = self._get_scale(P2)
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                output_logits = self.convs[("dispconv", i)](x)
                outputs[('logits', i)] = output_logits
                outputs[('depth', i, i)], outputs[('disp', i)] = self.gather_output(output_logits, depth_scale)
        return outputs


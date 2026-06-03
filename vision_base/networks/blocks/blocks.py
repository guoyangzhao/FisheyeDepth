import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from typing import Optional, Callable
from vision_base.utils.utils import torch_assert as _assert
from vision_base.networks.utils.format import Format, nchw_to
from enum import Enum

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class GradientScale(nn.Module):
    """ Scale the gradient of the output while keeping the output value unchanged
    """
    def __init__(self, scale=1.0):
        super(GradientScale, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        return x * self.scale + x.detach() * (1 - self.scale)

class DeDict(nn.Module):
    """
    """

    def __init__(self, name='features'):
        super(DeDict, self).__init__()
        self.name = name
        
    def forward(self, x):
        return x[self.name]

class ConvBnReLU(nn.Module):
    """Some Information about ConvBnReLU"""

    def __init__(self, input_features=1, output_features=1, kernel_size=(1, 1), stride=[1, 1], padding='SAME', dilation=1, groups=1, relu=True, **kwargs):
        super(ConvBnReLU, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        pad_num = int((kernel_size[0] - 1) / 2) * \
            dilation if padding.lower() == 'same' else 0
        self.sequence = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=kernel_size,
                      stride=stride, padding=pad_num, dilation=dilation, groups=groups, **kwargs),
            nn.BatchNorm2d(output_features),
        )
        self.relu=True

    def forward(self, x):
        x = self.sequence(x)
        if self.relu:
            return F.relu(x)
        else:
            return x


class ConvReLU(nn.Module):
    """Some Information about ConvReLU"""

    def __init__(self, input_features=1, output_features=1, kernel_size=(1, 1), stride=[1, 1], padding='SAME'):
        super(ConvReLU, self).__init__()
        pad_num = int((kernel_size[0] - 1) / 2) if padding.lower() == 'same' else 0
        self.sequence = nn.Sequential(
            nn.Conv2d(input_features, output_features,
                      kernel_size=kernel_size, stride=stride, padding=pad_num),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvELU(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvELU, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class LinearLNDropout(nn.Module):
    """Some Information about LinearLNDropoutReLU"""

    def __init__(self, input_features=1,  num_hiddens=1, drop=0.0):
        super(LinearLNDropout, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_features, num_hiddens),
            nn.LayerNorm(num_hiddens),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x

class LinearBnReLU(nn.Module):
    """Some Information about LinearBnReLU"""

    def __init__(self, input_features=1,  num_hiddens=1):
        super(LinearBnReLU, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_features, num_hiddens),
            nn.GroupNorm(16, num_hiddens),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


class LinearDropoutReLU(nn.Module):
    """Some Information about LinearDropoutReLU"""

    def __init__(self, input_features=1,  num_hiddens=1, drop=0.0):
        super(LinearDropoutReLU, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_features, num_hiddens),
            nn.Dropout(drop),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


class FocalLoss(nn.Module):
    """Some Information about FocalLoss"""

    def __init__(self, alpha, weights):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weights, reduction='none')

    def forward(self, scores, probs, targets):
        label_weights = torch.ones_like(scores[:, 0])
        bg_inds = targets == 0
        fg_inds = targets == 1
        label_weights[bg_inds] *= probs[bg_inds, 0] ** self.alpha
        label_weights[fg_inds] *= probs[fg_inds, 1] ** self.alpha

        base_loss = self.cross_entropy(scores, targets)
        return (base_loss * label_weights).mean(), label_weights


class ModifiedSmoothedL1(nn.Module):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                     |x| - 0.5 / sigma^2,    otherwise
    '''

    def __init__(self, sigma):
        super(ModifiedSmoothedL1, self).__init__()
        self.sigma2 = sigma * sigma

    def forward(self, deltas, targets, sigma=None):
        sigma2 = self.sigma2 if sigma is None else sigma * sigma
        diffs = deltas - targets

        option1 = diffs * diffs * 0.5 * sigma2
        option2 = torch.abs(diffs) - 0.5 / sigma2
        
        condition_for_1 = (diffs < (1.0/sigma2)).float()
        smooth_l1 = option1 * condition_for_1 + option2 * (1-condition_for_1)
        return smooth_l1


class UpsampleFPN(nn.Module):
    """
        structures for retianets
    """
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(UpsampleFPN, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x]

class PyramidFeatures(nn.Module):
    """
        structures for retianets
    """
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        return [P3_x, P4_x, P5_x, P6_x] #, P7_x]


class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 4, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv2(x)

        return x

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1, 3, 4)  # B,D,C,H,W --> B,C,D,H,W
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)

        x = self.prob(x)
        x = x.squeeze(1)  # B D H W

        return x  # B D H W

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def localmax(cost_prob, radius, casbin, depth_bins):
    pred_idx = torch.argmax(cost_prob, 1, keepdim=True).float()  # B 1 H W
    pred_idx_low = pred_idx - radius
    pred_idx = torch.arange(0, 2*radius+1, 1, device=pred_idx.device).reshape(1, 2*radius+1,1,1).float()
    pred_idx = pred_idx + pred_idx_low  # B M H W
    pred_idx = torch.clamp(pred_idx, 0, casbin-1)
    pred_idx = pred_idx.long()
    depth = 0
    cost_prob_sum = 1e-6
    for i in range(2*radius+1):
        cost_prob_ = torch.gather(cost_prob, 1, pred_idx[:,i:i+1])
        depth = depth + torch.gather(depth_bins, 1, pred_idx[:, i:i+1])*cost_prob_
        cost_prob_sum = cost_prob_sum+cost_prob_
    depth = depth.div_(cost_prob_sum)
    return depth

class convex_upsample_layer(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super(convex_upsample_layer, self).__init__()
        self.scale = scale
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, (2**scale)**2*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )

    def forward(self, depth, feat):
        mask = self.upsample_mask(feat)
        return convex_upsample(depth, mask, self.scale)  # B H2 W2

def convex_upsample(depth, mask, scale=2):
    if len(depth.shape) == 3:
        B, H, W = depth.shape
        depth = depth.unsqueeze(1)
    else:
        B, _, H, W = depth.shape
    mask = mask.view(B, 9, 2**scale, 2**scale, H, W)
    mask = torch.softmax(mask, dim=1)

    up_ = F.unfold(F.pad(depth, [1, 1, 1, 1], mode='reflect'), [3,3], padding=0)
    up_ = up_.view(B, 9, 1, 1, H, W)

    up_ = torch.sum(mask * up_, dim=1)  # B, 2**scale, 2**scale, H, W
    up_ = up_.permute(0, 3, 1, 4, 2)  # B H 2**scale W 2**scale
    return up_.reshape(B, 2**scale*H, 2**scale*W)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).
    Args:
        drop_prob (float): Drop rate for paths of model. Dropout rate has
            to be between 0 and 1. Default: 0.
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0], ) + (1, ) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output

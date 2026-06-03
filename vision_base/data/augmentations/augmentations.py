"""
This file contains all PyTorch data augmentation functions.

Every transform should have a __call__ function which takes in (self, image, imobj)
where imobj is an arbitary dict containing relevant information to the image.

In many cases the imobj can be None, which enables the same augmentations to be used
during testing as they are in training.

Optionally, most transforms should have an __init__ function as well, if needed.
"""

import numpy as np
from numpy import random
from typing import Optional
import torch
import cv2
from vision_base.utils.builder import Sequential
from vision_base.data.augmentations.utils import flip_relative_pose

class EmptyAug(object):
    """
    Converts image data type to float.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, data):
        return data

class ExtractData(object):
    """ Extract Certain keys from data. Trimmed the output of different datasets to a unified minimum output.
        extract_keys : List[str], list of keys to be extracted
        mapped_keys  : Dict[str, str], a dictionary of keys mapping to new names

        We requires that all keys are presented in the input.
    """
    def __init__(self, extract_keys=[], mapped_keys={}):
        self.extract_keys = extract_keys
        self.mapped_keys = mapped_keys

    def __call__(self, data):
        output_data = dict()
        for key in self.extract_keys:
            output_data[key] = data[key]
        
        for key in self.mapped_keys:
            output_data[self.mapped_keys[key]] = data[key]
        return output_data

class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __init__(self, image_keys=['image'], **kwargs):
        self.image_keys = image_keys

    def __call__(self, data):
        for key in self.image_keys:
            data[key] = data[key].astype(np.float32)
        return data

class ConvertToTensor(object):
    """
    Converts image to tensor, shuffle channel.
    """
    def __init__(self, image_keys=['image'],
                       gt_image_keys=[],
                       calib_keys=[],
                       lidar_keys=[],
                       **kwargs):
        self.image_keys = image_keys
        self.gt_image_keys = gt_image_keys
        self.calib_keys = calib_keys
        self.lidar_keys = lidar_keys

    def __call__(self, data):
        for key in (self.image_keys + self.gt_image_keys):
            if len(data[key].shape) == 3:
                data[key] = torch.tensor(data[key].transpose([2, 0, 1]), dtype=torch.float32).contiguous()
            else:
                data[key] = torch.tensor(data[key]).contiguous()
        
        for key in self.calib_keys:
            data[key] = torch.tensor(data[key], dtype=torch.float32).contiguous()

        for key in self.lidar_keys:
            data[key] = torch.tensor(data[key], dtype=torch.float32)

        return data

class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds, image_keys=['image'], **kwargs):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.image_keys = image_keys

    def __call__(self, data):
        for key in self.image_keys:
            image = data[key]
            image = image.astype(np.float32)
            image /= 255.0
            image -= np.tile(self.mean, int(image.shape[2]/self.mean.shape[0]))
            image /= np.tile(self.stds, int(image.shape[2]/self.stds.shape[0]))
            image.astype(np.float32)
            data[key] = image
        return data


class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size, preserve_aspect_ratio=True,
                       force_pad=True,
                       image_keys=['image'],
                       calib_keys=[],
                       gt_image_keys=[],
                        **kwargs):
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.force_pad = force_pad
        self.image_keys=image_keys
        self.calib_keys=calib_keys
        self.gt_image_keys=gt_image_keys

    def __call__(self, data):
        image = data[self.image_keys[0]]
        data[('image_resize', 'original_shape')] = np.array([image.shape[0], image.shape[1]]).astype(np.int32)
        ## Set up reshape output
        if self.preserve_aspect_ratio:
            scale_factor_x = self.size[0] / image.shape[0]
            scale_factor_y = self.size[1] / image.shape[1]
            if self.force_pad:
                scale_factor = min(scale_factor_x, scale_factor_y)
                mode = 'pad_0' if scale_factor_x > scale_factor_y else 'pad_1'
            else:
                scale_factor = scale_factor_x
                mode = 'crop_1' if scale_factor_x > scale_factor_y else 'pad_1'

            h = np.round(image.shape[0] * scale_factor).astype(int)
            w = np.round(image.shape[1] * scale_factor).astype(int)
            
            scale_factor_yx = (scale_factor, scale_factor)
        else:
            scale_factor_yx = (self.size[0] / image.shape[0], self.size[1] / image.shape[1])
            mode = 'none'
            h = self.size[0]
            w = self.size[1]
        
        data[('image_resize', 'effective_size')] = np.array([h, w]).astype(np.int32)

        # resize
        for key in self.image_keys:
            data[key] = cv2.resize(data[key], (w, h))

        for key in self.gt_image_keys:
            data[key] = cv2.resize(data[key], (w, h), interpolation=cv2.INTER_NEAREST)


        if len(self.size) > 1:

            for key in (self.image_keys + self.gt_image_keys):
                image = data[key]
                # crop in
                if mode=='crop_1':
                    data[key] = image[:, 0:self.size[1]]
                
                # pad
                if mode == 'pad_1':
                    padW = self.size[1] - image.shape[1]
                    if len(image.shape) == 2:
                        data[key] = np.pad(image,  [(0, 0), (0, padW)], 'constant')
                    
                    elif len(image.shape) == 3:
                        data[key] = np.pad(image,  [(0, 0), (0, padW), (0, 0)], 'constant')

                if mode == 'pad_0':
                    padH = self.size[0] - image.shape[0]
                    if len(image.shape) == 2:
                        data[key] = np.pad(image,  [(0, padH), (0, 0)], 'constant')
                    
                    elif len(image.shape) == 3:
                        data[key] = np.pad(image,  [(0, padH), (0, 0), (0, 0)], 'constant')

        for key in self.calib_keys:
            P = data[key]
            P[0, :] = P[0, :] * scale_factor_yx[1]
            P[1, :] = P[1, :] * scale_factor_yx[0]
            data[key] = P

        return data

class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5, image_keys=['image'], random_seed=None, **kwargs):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

        self.image_keys = image_keys
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):
        if self.rng.random() <= self.distort_prob:
            ratio = self.rng.uniform(self.lower, self.upper)

            for key in self.image_keys:
                data[key][:, :, 1] *= ratio

        return data

class CropTop(object):
    def __init__(self, crop_top_index=None, output_height=None,
                image_keys=['image'],
                gt_image_keys=[],
                calib_keys=[],
                **kwargs
                ):
        if crop_top_index is None and output_height is None:
            print("Either crop_top_index or output_height should not be None, set crop_top_index=0 by default")
            crop_top_index = 0
        if crop_top_index is not None and output_height is not None:
            print("Neither crop_top_index or output_height is None, crop_top_index will take over")
        self.crop_top_index = crop_top_index
        self.output_height = output_height
        self.image_keys = image_keys
        self.calib_keys = calib_keys
        self.gt_image_keys = gt_image_keys

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]

        if self.crop_top_index is not None:
            upper = self.crop_top_index
        else:
            upper = height - self.output_height
        lower = height

        for key in (self.image_keys + self.gt_image_keys):
            data[key] = data[key][upper:lower]

        ## modify calibration matrix
        for key in self.calib_keys:
            P = data[key]
            P[1, 2] = P[1, 2] - upper               # cy' = cy - dv
            P[1, 3] = P[1, 3] - upper * P[2, 3] # ty' = ty - dv * tz
            data[key] = P
        
        return data


class CropRight(object):
    def __init__(self, crop_right_index=None, output_width=None,
                image_keys=['image'],
                gt_image_keys=[],
                **kwargs):
        if crop_right_index is None and output_width is None:
            print("Either crop_right_index or output_width should not be None, set crop_right_index=0 by default")
            crop_right_index = 0
        if crop_right_index is not None and output_width is not None:
            print("Neither crop_right_index or output_width is None, crop_right_index will take over")
        self.crop_right_index = crop_right_index
        self.output_width = output_width
        self.gt_image_keys = gt_image_keys

    def __call__(self, data):

        height, width = data[self.image_keys[0]].shape[0:2]

        lefter = 0
        if self.crop_right_index is not None:
            w_out = width - self.crop_right_index
            righter = w_out
        else:
            w_out = self.output_width
            righter = w_out
        
        if righter > width:
            print("does not crop right since it is larger")
            return data

        for key in (self.image_keys + self.gt_image_keys):
            data[key] = data[key][:, lefter:righter]

        return data


class Pad2Shape(object):
    def __init__(self, target_shape, image_keys=['image'],
                gt_image_keys=[],
                **kwargs):
        self.target_shape = target_shape
        self.image_keys = image_keys
        self.gt_image_keys = gt_image_keys

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]
        for key in self.image_keys + self.gt_image_keys:
            padH = self.target_shape[0] - height
            padW = self.target_shape[1] - width
            image = data[key]
            if len(image.shape) == 2:
                data[key] = np.pad(image,  [(0, padH), (0, padW)], 'constant')
            
            elif len(image.shape) == 3:
                data[key] = np.pad(image,  [(0, padH), (0, padW), (0, 0)], 'constant')

        return data


class FilterObject(object):
    """
        Filtering out object completely outside of the box;
    """
    def __init__(self, image_keys=['image'], objects_keys=[], **kwargs):
        self.image_keys = image_keys
        self.object_keys = objects_keys

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]

        for key in self.object_keys:
            data[key].filter(height, width)
        
        return data

class RandomCropToWidth(object):
    def __init__(self, width:Optional[int]=None,
                 width_ratio_range:Optional[tuple]=None,
                image_keys=['image'],
                gt_image_keys=[],
                calib_keys=[],):
        self.width = width
        self.width_ratio_range = width_ratio_range
        assert self.width is not None or self.width_ratio_range is not None, \
              "Either width or width_ratio_range should not be None"
        self.image_keys = image_keys
        self.calib_keys = calib_keys
        self.gt_image_keys = gt_image_keys

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]
        original_width = width

        if self.width is not None:
            output_width = self.width
        elif self.width_ratio_range is not None:
            output_width = int(np.random.uniform(*self.width_ratio_range) * original_width)
        else:
            raise NotImplementedError
        
        if output_width > original_width:
            print("does not crop since it is larger")
            return data
        lefter = np.random.randint(0, original_width - output_width)
        righter = lefter + output_width


        for key in (self.image_keys + self.gt_image_keys):
            data[key] = data[key][:, lefter:righter]

        ## modify calibration matrix
        for key in self.calib_keys:
            P = data[key]
            P[0, 2] = P[0, 2] - lefter               # cy' = cy - dv
            P[0, 3] = P[0, 3] - lefter * P[2, 3] # ty' = ty - dv * tz
            data[key] = P

        return data

class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty. It will also flip world in 3D

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob,
                image_keys=['image'],
                calib_keys=[],
                gt_image_keys=[],
                object_keys=[],
                lidar_keys=[],
                pose_axis_pairs=[],
                is_switch_left_right=True,
                stereo_image_key_pairs=[], #only used in "is_switch_left_right==True"
                stereo_calib_key_pairs=[], #only used in "is_switch_left_right==True"
                **kwargs
                ):
        self.mirror_prob = mirror_prob
        self.image_keys = image_keys
        self.calib_keys = calib_keys
        self.gt_image_keys = gt_image_keys
        self.object_keys = object_keys
        self.lidar_keys = lidar_keys
        self.pose_axis_pairs  = pose_axis_pairs
        self.is_switch_lr = is_switch_left_right
        self.stereo_image_key_pairs = stereo_image_key_pairs
        self.stereo_calib_key_pairs = stereo_calib_key_pairs

    def __call__(self, data):

        height, width, _ = data[self.image_keys[0]].shape

        if random.rand() <= self.mirror_prob:
            
            for key in (self.image_keys + self.gt_image_keys):
                data[key] = np.ascontiguousarray(data[key][:, ::-1])

            for key in self.calib_keys:
                P = data[key]
                P[0, 3] = -P[0, 3]
                P[0, 2] = width - P[0, 2] - 1
                data[key] = P

            for key in self.object_keys:
                data[key].flip_objects()

            for key in self.lidar_keys:
                data[key] = -data[key][..., 0] # Assume the last channel in lidar is "x" in the width direction
            
            for key, axis_num in self.pose_axis_pairs:
                data[key] = flip_relative_pose(data[key], axis_num)

            if self.is_switch_lr:
                for key_l, key_r in (self.stereo_image_key_pairs + self.stereo_calib_key_pairs):
                    data[key_l], data[key_r] = data[key_r], data[key_l]
        
        return data

class RandomWarpAffine(object):
    """
        Randomly random scale and random shift the image. Then resize to a fixed output size.
    """
    def __init__(self, scale_lower=0.6, scale_upper=1.4, shift_border=128, output_w=1280, output_h=384,
                image_keys=['image'],
                gt_image_keys=[],
                calib_keys=[],
                border_mode=cv2.BORDER_CONSTANT,
                random_seed = None,
                **kwargs):
        self.scale_lower    = scale_lower
        self.scale_upper    = scale_upper
        self.shift_border   = shift_border
        self.output_w       = output_w
        self.output_h       = output_h

        self.image_keys = image_keys
        self.gt_image_keys = gt_image_keys
        self.calib_keys = calib_keys
        self.border_mode = border_mode

        # If we provide a fix random seed, for different augmentation instances, they will provide random warp in the same way.
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):
        height, width = data[self.image_keys[0]].shape[0:2]

        s_original = max(height, width)
        scale = s_original * self.rng.uniform(self.scale_lower, self.scale_upper)
        center_w = self.rng.integers(low=self.shift_border, high=width - self.shift_border)
        center_h = self.rng.integers(low=self.shift_border, high=height - self.shift_border)

        final_scale = max(self.output_w, self.output_h) / scale
        final_shift_w = self.output_w / 2 - center_w * final_scale
        final_shift_h = self.output_h / 2 - center_h * final_scale
        affine_transform = np.array(
            [
                [final_scale, 0, final_shift_w],
                [0, final_scale, final_shift_h]
            ], dtype=np.float32
        )

        for key in self.image_keys:
            data[key] = cv2.warpAffine(
                data[key], affine_transform, (self.output_w, self.output_h), flags=cv2.INTER_LINEAR, borderMode=self.border_mode
            )

        for key in self.gt_image_keys:
            data[key] = cv2.warpAffine(
                data[key], affine_transform, (self.output_w, self.output_h), flags=cv2.INTER_NEAREST, borderMode=self.border_mode
            )

        for key in self.calib_keys:
            P = data[key]
            P[0:2, :] *= final_scale
            P[0, 2] = P[0, 2] + final_shift_w               # cy' = cy - dv
            P[0, 3] = P[0, 3] + final_shift_w * P[2, 3] # ty' = ty - dv * tz
            P[1, 2] = P[1, 2] + final_shift_h               # cy' = cy - dv
            P[1, 3] = P[1, 3] + final_shift_h * P[2, 3] # ty' = ty - dv * tz
            data[key] = P

        return data

class RandomDistortion(object):
    """
    Randomly apply Camera distortion to an image given a distortion probability.
    """
    def __init__(self, distort_prob, image_keys=['image'], calib_keys=[], distortion_base_strength=3.0,
                 gt_image_keys=[], random_seed=None, **kwargs):
        self.distort_prob = distort_prob
        self.image_keys = image_keys
        self.calib_keys = calib_keys
        self.gt_image_keys = gt_image_keys
        self.distort_base = distortion_base_strength
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))
    
    def __call__(self, data):
        if self.rng.random() <= self.distort_prob:
            
            D = self.rng.normal(0, 1, 4) * np.array([(self.distort_base * 0.1)**i for i in range(4)])
            for key in (self.image_keys + self.gt_image_keys):
                original_image = data[key]
                if len(self.calib_keys):
                    K = data[self.calib_keys[0]][0:3, 0:3]
                else:
                    h, w = original_image.shape[:2]
                    K = np.array([[w, 0, w/2],
                              [0, w, h/2],
                              [0, 0, 1]], dtype=np.float32)
                mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
                data[key] = cv2.remap(original_image, mapx, mapy, cv2.INTER_NEAREST)
        
        return data

class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0, image_keys=['image'], random_seed=None, **kwargs):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob
        self.image_keys = image_keys
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):
        if self.rng.random() <= self.distort_prob:
            shift = self.rng.uniform(-self.delta, self.delta)
            for key in self.image_keys:
                image = data[key]
                image[:, :, 0] += shift
                image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
                image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
                data[key] = image

        return data


class ConvertColor(object):
    """
    Converts color spaces to/from HSV and RGB
    """
    def __init__(self, current='RGB', transform='HSV', image_keys=['image'], **kwargs):
        self.transform = transform
        self.current = current
        self.image_keys = image_keys
        self.convertor = getattr(cv2, f"COLOR_{current}2{transform}")

    def __call__(self, data):
        
        for key in self.image_keys:
            data[key] = cv2.cvtColor(data[key], self.convertor)

        return data


class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5, image_keys=['image'], random_seed=None, **kwargs):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        self.image_keys = image_keys

        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, data):
        if self.rng.random() <= self.distort_prob:
            alpha = self.rng.uniform(self.lower, self.upper)

            for key in self.image_keys:
                data[key] = data[key] * alpha

        return data

class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32, image_keys=['image'], random_seed=None, **kwargs):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob
        self.image_keys = image_keys
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):
        if self.rng.random() <= self.distort_prob:
            delta = self.rng.uniform(-self.delta, self.delta)

            for key in self.image_keys:
                data[key] = data[key] + delta

        return data

class RandomEigenvalueNoise(object):
    """
        Randomly apply noise in RGB color channels based on the eigenvalue and eigenvector of ImageNet
    """
    def __init__(self, distort_prob=1.0,
                       alphastd=0.1,
                       eigen_value=np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
                       eigen_vector=np.array([
                            [-0.58752847, -0.69563484, 0.41340352],
                            [-0.5832747, 0.00994535, -0.81221408],
                            [-0.56089297, 0.71832671, 0.41158938]
                        ], dtype=np.float32),
                        image_keys=['image'],
                        random_seed=None,
                        **kwargs,
                ):
        self.distort_prob = distort_prob
        self._eig_val = eigen_value
        self._eig_vec = eigen_vector
        self.alphastd = alphastd
        self.image_keys = image_keys
        self.rng = np.random.default_rng(random_seed if random_seed is not None else np.random.randint(0, 2**32))

    def __call__(self, data):

        if self.rng.random() <= self.distort_prob:
            alpha = self.rng.normal(scale=self.alphastd, size=(3, ))
            noise = np.dot(self._eig_vec, self._eig_val * alpha) * 255

            for key in self.image_keys:
                data[key] = data[key] + noise
            
        return data

class PhotometricDistort(object):
    """
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32, image_keys=['image'], **kwargs):

        self.distort_prob = distort_prob

        # contrast is duplicated because it may happen before or after
        # the other transforms with equal probability.
        self.transforms = [
            RandomContrast(distort_prob, contrast_lower, contrast_upper, image_keys=image_keys),
            ConvertColor(transform='HSV', image_keys=image_keys),
            RandomSaturation(distort_prob, saturation_lower, saturation_upper, image_keys=image_keys),
            RandomHue(distort_prob, hue_delta, image_keys=image_keys),
            ConvertColor(current='HSV', transform='RGB', image_keys=image_keys),
            RandomContrast(distort_prob, contrast_lower, contrast_upper, image_keys=image_keys)
        ]

        self.rand_brightness = RandomBrightness(distort_prob, brightness_delta, image_keys=image_keys)

    def __call__(self, data):

        # do contrast first
        if random.rand() <= 0.5:
            distortion = self.transforms[:-1]

        # do contrast last
        else:
            distortion = self.transforms[1:]

        # add random brightness
        distortion.insert(0, self.rand_brightness)

        # compose transformation
        distortion_list = Sequential(cfg_list=[])
        distortion_list.children = distortion

        return distortion_list(data)

class Copy(object):
    """
    Copy the image to another key
    """
    def __init__(self, from_keys, to_keys, **kwargs):
        self.from_keys = from_keys
        self.to_keys = to_keys

    def __call__(self, data):
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            data[to_key] = data[from_key].copy()

        return data

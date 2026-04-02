import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from vision_base.networks.blocks.blocks import ConvBnReLU
from monodepth.networks.utils.monodepth_utils import SSIM

def build_inv_K(P):
    K = P[:, 0:3, 0:3]
    inv_K = torch.inverse(K)
    return inv_K

def project_on_image(points, P):
    """ points: [B, *, 3]
        P: [B, 3, 4]
    """
    homo_points = torch.cat([points,
                torch.ones_like(points[..., 0:1])],
                dim=-1) #[B, *, 4]
    dim_length = len(homo_points.shape) - 2
    for i in range(dim_length):
        P = P.unsqueeze(1)
    image_pts = torch.matmul(P, homo_points.unsqueeze(-1))[..., 0] # [B, *, 3]
    points = image_pts[..., 0:2] / image_pts[..., 2:3]
    return torch.cat([points, image_pts[..., 2:3]], dim=-1) #[cam_x, cam_y, Z]

def transform_points(points, T):
    """ points: [B, *, 3]
        T : [B, 4, 4]
    """
    homo_points = torch.cat([points,
                torch.ones_like(points[..., 0:1])],
                dim=-1) #[B, *, 4]
    dim_length = len(homo_points.shape) - 2
    for i in range(dim_length):
        T = T.unsqueeze(1)
    transformed_pts = torch.matmul(T, homo_points.unsqueeze(-1))[..., 0] # [B, *, 4]
    return transformed_pts[..., 0:3] #[B, *, 3]
    


class OccupancyHead(nn.Module):
    def __init__(self, input_features,
                 input_feature_lengths=120,
                 output_ch=4,
                num_transformers=3, pts_batches=64,
                embedding_feature=256,
                render_sample_range=[0.5, 40, 0.5],
                x_range=[-20, 20, 0.4],
                y_range=[-2, 3, 0.5],
                z_range=[2, 42, 0.4],
                down_scale=2,
                raw_noise_std=0.0,
                num_heads=8,
                train_cfg=dict(),
                **kwargs):
        super().__init__()
        self.input_features = input_features
        self.input_feature_length = input_feature_lengths
        self.output_channel = output_ch
        self.embedding_feature = embedding_feature
        self.pts_batches = pts_batches
        self.num_transformers = num_transformers
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_heads = num_heads
        self.down_scale = down_scale
        
        self.render_sample_range = render_sample_range
        self.register_buffer('render_z_range', torch.arange(self.render_sample_range[0], self.render_sample_range[1], self.render_sample_range[2]))
        dists = self.render_z_range[1:] - self.render_z_range[:-1] # [Z - 1]
        dists = torch.cat([dists, torch.Tensor([1e10])])
        self.register_buffer('base_dist', dists) #[Z]
        self.raw_noise_std = raw_noise_std


        self.train_cfg = train_cfg

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._init_layers()

    def _init_layers(self):
        self.size_X = int((self.x_range[1] - self.x_range[0]) / self.x_range[2])
        self.size_Y = int((self.y_range[1] - self.y_range[0]) / self.y_range[2])
        self.size_Z = int((self.z_range[1] - self.z_range[0]) / self.z_range[2])
        self.embedding = nn.Embedding((self.size_X * self.size_Y * self.size_Z) // (self.down_scale) ** 3,
                                       self.embedding_feature)
        self.camera_embedding = nn.Embedding((self.input_feature_length), self.input_features)
        self.key_linear = nn.Linear(self.input_features, self.embedding_feature)
        self.value_linear = nn.Linear(self.input_features, self.embedding_feature)
        self.transformers = nn.ModuleList(
            [nn.MultiheadAttention(self.embedding_feature, self.num_heads, batch_first=True) for _ in range(self.num_transformers)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embedding_feature, self.embedding_feature),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embedding_feature, self.embedding_feature),
                    nn.LayerNorm(normalized_shape=self.embedding_feature),
                    nn.Dropout(p=0.1),
                ) for _ in range(self.num_transformers)
            ]
        )
        self.upsample = nn.Sequential(
            nn.Conv3d(self.embedding_feature, self.embedding_feature, 3, padding=1),
            nn.BatchNorm3d(num_features=self.embedding_feature),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.embedding_feature, self.embedding_feature, 3, padding=1),
            nn.BatchNorm3d(num_features=self.embedding_feature),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=self.down_scale, mode="trilinear")
        )
        self.output_proj = nn.Sequential(
            nn.Conv3d(self.embedding_feature, self.output_channel, 1),
        )
        self.ssim = SSIM()


    def forward(self, features, *args, **kwargs):
        f = features[-1]
        B, C, H, W = f.shape
        assert C == self.input_features
        flatten_features = f.view(B, self.input_features, -1).permute(0, 2, 1) +\
              self.camera_embedding.weight.unsqueeze(0) # [B, N, C]
        key = self.key_linear(flatten_features) # [B, N, C]
        value = self.value_linear(flatten_features) # [B, N, C]
        query = self.embedding.weight.unsqueeze(0).repeat(B, 1, 1) # [B, N, C]

        for i in range(self.num_transformers):
            out, _ = self.transformers[i](query, key, value) # [B, N, C]
            out = out + self.ffns[i](out)
            query = out + self.embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        output = query.view(B, self.size_X//self.down_scale, self.size_Y//self.down_scale, self.size_Z//self.down_scale, self.embedding_feature).permute(0, 4, 1, 2, 3) # [B, C, X, Y, Z]
        output = self.upsample(output) # [B, C, X, Y, Z]
        output = self.output_proj(output) # [B, 4, X, Y, Z]

        return dict(output_voxel=output)

    def extract_output_from_pts(self, target_pts, output_dict):
        x = target_pts[..., 0] # [B, 2, Z, pts_batches]
        y = target_pts[..., 1]
        z = target_pts[..., 2]
        x_norm_index = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        y_norm_index = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        z_norm_index = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 2 - 1
        grid_3d_normed = torch.stack([z_norm_index, y_norm_index, x_norm_index], -1) # [B, 2, Z, pts_batches, 3]

        voxel = output_dict['output_voxel'] # [B, 4, X, Y, Z]
        extracted_output = F.grid_sample(voxel, grid_3d_normed, align_corners=True, padding_mode='zeros') # [B, 4, 2, Z, pts_batches]
        return extracted_output

    def render(self, random_pix_x, random_pix_y, output_dict, input_dict, train_idxs):
        # random_pix_x : [B, 2, pts_batches]
        homo_random_pix = torch.stack([random_pix_x,
                                       random_pix_y,
                                       torch.ones_like(random_pix_x)], 2) # [B, 2, 3, pts_batches]
        inv_K = build_inv_K(input_dict['P2']).unsqueeze(1) # [B, 1, 3, 3]

        cam_pts_base = torch.matmul(inv_K, homo_random_pix) # [B, 2, 3, pts_batches]

        # dists : [B, 2, Z, pts_batches]
        dists = torch.norm(cam_pts_base, dim=2, keepdim=True) * self.base_dist.reshape([1, 1, -1, 1])
        
        # [B, 2, Z, 3, pts_batches]
        cam_pts  = cam_pts_base.unsqueeze(2) * self.render_z_range.reshape([1, 1, -1, 1, 1])
        homo_cam_pts = torch.cat(
            [cam_pts, torch.ones_like(cam_pts[:, :, :, 0:1, :])], dim=-2
        ) # [B, 2, Z, 4, pts_batches]

        T_target2src = torch.stack(
            [input_dict[('relative_pose', f_i)] for f_i in train_idxs], dim=1
        ).unsqueeze(2) #[B, 2, 1, 4, 4]
        T_src2target = torch.inverse(T_target2src) #[B, 2, 1, 4, 4]

        target_pts = torch.matmul(T_src2target,
                                 homo_cam_pts).permute(0, 1, 2, 4, 3) # [B, 2, Z, pts_batches, 4]
        raw = self.extract_output_from_pts(target_pts, output_dict).permute(0, 2, 3, 4, 1) #[B, 2, Z, pts_batches, num_feats]
        seg = raw[..., 1:] #[B, 2, Z, pts_batches, C]
        objectness = torch.nn.functional.relu(raw[..., 0] + torch.randn(raw[..., 0].shape, device=raw.device) * self.raw_noise_std)
        alpha = 1 - torch.exp(-objectness * dists) #[B, 2, Z, pts_batches]
        weights = alpha * torch.cumprod(
            torch.cat(
                [torch.ones_like(alpha[:, :, 0:1, ]), 1-alpha + 1e-10], dim=2
            ), dim=2
        )[:, :, :-1, ...] # [B, 2, Z, pts_batches]
        seg_map = torch.sum(weights[..., None] * seg, dim=-3) #[B, 2, pts_batches, C]
        depth_map = torch.sum(weights * self.render_z_range[:, None], dim=-2)
        return dict(seg_map=seg_map, depth_map=depth_map)
    
    def get_prediction(self, input_dict, output_dict):
        # rgb = torch.sigmoid(output_dict['output_voxel'][:, 0:3]) #[B, 3, X, Y, Z]
        # rgb = output_dict['output_voxel'][:, 0:3] #[B, 3, X, Y, Z]
        # occupancy = torch.nn.functional.relu(output_dict['output_voxel'][:, 3]) #[B, X,Y,Z]
        # objectness = torch.nn.functional.relu(output_dict['output_voxel'][:, 3])
        objectness = torch.nn.functional.relu(output_dict['output_voxel'][:, 0])
        occupancy = torch.clamp(1 - torch.exp(-objectness * self.z_range[2]), 1e-5, 1 - 1e-5)
        segmentation = torch.argmax(output_dict['output_voxel'][:, 1:], dim=1)
        coordinates = torch.stack(torch.meshgrid(
            torch.arange(self.x_range[0], self.x_range[1], self.x_range[2]),
            torch.arange(self.y_range[0], self.y_range[1], self.y_range[2]),
            torch.arange(self.z_range[0], self.z_range[1], self.z_range[2]),
        ), dim=-1).cuda()# [X, Y, Z]
        # return dict(rgb=rgb, occupancy=occupancy, coordinates=coordinates)
        return dict(occupancy=occupancy, coordinates=coordinates, segmentation=segmentation)

    def extract_target(self, random_pix_x, random_pix_y, input_dict,  train_idxs, keys=[('original_image', 0)]):
        target_pts = []
        for i, idx in enumerate(train_idxs):
            target_image = input_dict[keys[i]] # [B, 3, H, W]
            if len(target_image.shape) == 3:
                target_image = target_image.unsqueeze(1) # [B, 1, H, W]
            B, _, H, W = target_image.shape
            pix_x = random_pix_x[:, i:i+1, :] / W * 2 - 1 # [B, 1, pts_batches]
            pix_y = random_pix_y[:, i:i+1, :] / H * 2 - 1
            pix = torch.stack([pix_x, pix_y], dim=-1) # [B, 1, pts_batches, 2]
            target_pt = F.grid_sample(target_image.float(), pix, align_corners=True, padding_mode='zeros', mode='nearest') # [B, 3, 1, pts_batches]
            target_pts.append(target_pt.squeeze(2).permute(0, 2, 1)) # [B, pts_batches, 3]
        return torch.stack(target_pts, dim=1) # [B, 2, pts_batches, 3]
    
    def get_coordinates(self):
        coordinates = torch.stack(torch.meshgrid(
            torch.arange(self.x_range[0], self.x_range[1], self.x_range[2]),
            torch.arange(self.y_range[0], self.y_range[1], self.y_range[2]),
            torch.arange(self.z_range[0], self.z_range[1], self.z_range[2]),
        ), dim=-1).cuda()# [X, Y, Z, 3]
        return coordinates

    def get_teacher_voxel(self, teacher_depth_map, output_dict, P2):
        target_depth_clamp_to_bins = ((teacher_depth_map - self.z_range[0]) / self.z_range[2]).long() * self.z_range[2] + self.z_range[0]

        depth_bins = torch.arange(self.z_range[0], self.z_range[1], self.z_range[2])
        self.depth_bins = depth_bins.cuda()

        reshaped_depth_bins = self.depth_bins.reshape(1, -1, 1, 1) # [1, Z, 1, 1]
        hit_index = torch.isclose(reshaped_depth_bins, target_depth_clamp_to_bins) # [B, Z, H, W]

        B, Z, H, W = 1, len(depth_bins), 192, 640
        target_logits = torch.zeros([B, Z, H, W]).cuda() # [B, Z, H, W]
        target_logits[hit_index] = 1

        image_coordinates = project_on_image(
            self.get_coordinates().unsqueeze(0), P2) # [B, X, Y, Z, 3]
        X_index = image_coordinates[..., 0]
        Y_index = image_coordinates[..., 1]
        Z_index = (image_coordinates[..., 2] - self.z_range[0]) / self.z_range[2] # [B, X, Y, Z]
        normed_x_index = X_index / W * 2 - 1 #[B, X, Y, Z]
        normed_y_index = Y_index / H * 2 - 1
        normed_z_index = Z_index / Z * 2 - 1
        _, X3d, Y3d, Z3d = normed_x_index.shape
        occupancy = F.grid_sample(
            target_logits.unsqueeze(1), #[B, 1, Z, H, W]
            torch.stack([normed_x_index, normed_y_index, normed_z_index], dim=-1), #[B, X, Y, Z, 3]
        ) # [B, 1, X, Y, Z]

        coordinates = torch.stack(torch.meshgrid(
            torch.arange(self.x_range[0], self.x_range[1], self.x_range[2]),
            torch.arange(self.y_range[0], self.y_range[1], self.y_range[2]),
            torch.arange(self.z_range[0], self.z_range[1], self.z_range[2]),
        ), dim=-1).cuda()# [X, Y, Z]
        return dict(occupancy=occupancy[:, 0], coordinates=coordinates)
    
    def reconstruct(self, output_dict, input_dict):
        frame_idxs = getattr(self, 'frame_ids', [0, 1, -1])
        base_img = input_dict[('original_image', 0)]
        B, _, H, W = base_img.shape

        train_idxs = [t for t in frame_idxs if t != 0]
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(base_img.device)
        j = j.t().to(base_img.device) # [H, W]
        x = i.reshape(1, 1, -1).repeat(B, len(train_idxs), 1)  # [B, 2, H*W]
        y = j.reshape(1, 1, -1).repeat(B, len(train_idxs), 1)  # [B, 2, H*W]
        with torch.no_grad():
            input_dict[('relative_pose', 0)] = torch.eye(4, device=base_img.device).unsqueeze(0).repeat(B, 1, 1) #
            rendered = self.render(x, y, output_dict, input_dict, train_idxs) # [B, 2, H*W, 3]
        results = dict()
        for i, idx in enumerate(train_idxs):
            results[('reconstructed_image', idx)] = rendered['rgb_map'][:, i, :].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return results
    
    def distillation_loss(self, logits, teacher_depth_map):
        """ logits: [B, Z, H, W]
            depth_map : [B, 1, H, W]
        """
        target_depth_clamp_to_bins = ((teacher_depth_map - self.z_range[0]) / self.z_range[2]).long() * self.z_range[2] + self.z_range[0]
        depth_max_larger = target_depth_clamp_to_bins + self.z_range[0]
        reshaped_depth_bins = self.depth_bins.reshape(1, -1, 1, 1) # [1, Z, 1, 1]
        hit_index = torch.isclose(reshaped_depth_bins, target_depth_clamp_to_bins) # [B, Z, H, W]
        larger_than_max  = (reshaped_depth_bins > depth_max_larger) # [B, Z, H, W]

        target_logits = torch.zeros_like(logits) # [B, Z, H, W]
        target_logits[hit_index] = 1
        mask = torch.ones_like(logits) # [B, Z, H, W]
        mask[larger_than_max] = 0.01

        objectness = torch.nn.functional.relu(logits)
        alpha = torch.clamp(1 - torch.exp(-objectness * self.z_range[2]), 1e-5, 1 - 1e-5)

        loss = F.binary_cross_entropy(alpha, target_logits, reduction='none') # [B, Z, H, W]
        loss = torch.where(
            hit_index,
            loss * 30,
            loss
        )
        loss = loss * mask
        return loss.mean()


    def voxel_loss(self, output_voxel, target_occupancy, mask):
        objectness = torch.nn.functional.relu(output_voxel)
        occupancy = torch.clamp(1 - torch.exp(-objectness * self.z_range[2]), 1e-5, 1 - 1e-5)

        loss = F.binary_cross_entropy(occupancy, target_occupancy.float(), reduction='none') # [B, X, Y, Z]
        loss = torch.where(
            target_occupancy,
            loss * 30,
            loss
        )
        loss = loss * mask
        return loss.mean()

    def semantic_loss(self, output_voxel, target_semantic):
        output_voxel = torch.clamp(output_voxel, -30, 30)
        _loss = F.cross_entropy(output_voxel, target_semantic, reduction='none', ignore_index=0)
        key_loss = _loss[target_semantic > 0]
        return key_loss.mean()

    def loss(self, output_dict, input_dict):
        frame_idxs = getattr(self, 'frame_ids', [0, 1, -1])
        base_img = input_dict[('original_image', 0)]
        B, _, H, W = base_img.shape

        x = torch.randint(0, W-1, (B, len(frame_idxs), self.pts_batches), device=base_img.device).float()
        y = torch.randint(0, H-1, (B, len(frame_idxs), self.pts_batches), device=base_img.device).float()
        input_dict[('relative_pose', 0)] = torch.eye(4, device=base_img.device).unsqueeze(0).repeat(B, 1, 1) #
        
        render_dict = self.render(x, y, output_dict, input_dict, frame_idxs)
        depths = self.extract_target(x, y, output_dict, frame_idxs,
                    keys=[('teacher_depth', idx, 0) for idx in frame_idxs]) # [B, idxs, N, 1]
        segs   = self.extract_target(x, y, input_dict, frame_idxs,
                    keys=[('semantics', idx) for idx in frame_idxs]).long() # [B, idxs, N, 1]
        loss_dict = dict()

        # ## RGB construction
        # losses = []
        # for i, idx in enumerate(frame_idxs):
        #     reconstructed_image = render_dict['rgb_map'][:, i].reshape(B, H//2, W, 3).permute(0, 3, 1, 2)
        #     target = input_dict[('original_image', idx)][:, :, H//2:H, :]
        #     loss = (reconstructed_image - target).abs().mean(1, True) * 0.15 + 0.85 * self.ssim(reconstructed_image, target).mean(1, True)
        #     losses.append(loss)
        #     loss_dict[f'rgb_loss_{idx}'] = loss.mean().detach()
        # rgb_losses = torch.cat(losses, dim=1)

        ## Depth Prediction
        reconstructed_depth = render_dict['depth_map'] #[B, idxs, N]
        depth_loss = (reconstructed_depth - depths.squeeze(-1)).abs().mean(1, True).mean()
        loss_dict[f'depth_loss'] = depth_loss.detach()

        ## Depth Prediction
        seg = render_dict['seg_map'].permute(0, 3, 1, 2) #[B, idxs, N, C] -> [B, C, idxs, N]
        seg_loss = F.cross_entropy(seg, segs[..., 0], ignore_index=0).mean()
        loss_dict[f'seg_loss'] = seg_loss.detach()


        # ## Distillation Prediction
        # dist_loss = self.distillation_loss(output_dict[('logits', 0)],
        #                                 output_dict[('teacher_depth', 0, 0)]).mean()

        ## Voxel loss
        voxel_loss = self.voxel_loss(output_dict['output_voxel'][:, 0],
                     input_dict['occupancy'], input_dict['voxel_mask'])
        semantic_loss = self.semantic_loss(output_dict['output_voxel'][:, 1:], input_dict['semantics'])

        loss_dict[f'voxel_loss'] = voxel_loss.detach()
        loss_dict[f'semantic_loss'] = semantic_loss.detach()

        total_loss = voxel_loss + semantic_loss + depth_loss * 0.2 + seg_loss * 1.0
        loss_dict['total_loss'] = total_loss.detach()
        return {'loss': total_loss, 'loss_dict':loss_dict}
    

class PVSplatHead(OccupancyHead):
    def _init_layers(self):
        self.base_fx = None
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_output_channels = int((self.z_range[1] - self.z_range[0]) / self.z_range[2])
        
        # self.processing = nn.Sequential(
        #     nn.Conv3d(4, self.embedding_feature, 3, padding=1),
        #     nn.BatchNorm3d(num_features=self.embedding_feature),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(2),
        #     nn.Conv3d(self.embedding_feature, self.embedding_feature, 3, padding=1),
        #     nn.BatchNorm3d(num_features=self.embedding_feature),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(self.embedding_feature, self.embedding_feature, 3, padding=1),
        #     nn.BatchNorm3d(num_features=self.embedding_feature),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(self.embedding_feature, self.embedding_feature, 3, padding=1),
        #     nn.BatchNorm3d(num_features=self.embedding_feature),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv3d(self.embedding_feature, 3, 3, padding=1),
        # )

        self._build_depth_bins()
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
            # y0 = 1 / self.num_output_channels
            # nn.init.constant_(self.convs[('dispconv', s)].bias, np.log(y0/(1-y0)))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.ssim = SSIM()


    def _build_depth_bins(self):
        depth_bins = torch.arange(self.z_range[0], self.z_range[1], self.z_range[2])
        self.register_buffer("depth_bins", depth_bins)

    def _gather_activation(self, x:torch.Tensor)->torch.Tensor:
        """Decode the output of the cost volume into a encoded depth feature map.

        Args:
            x (torch.Tensor): The output of the cost volume of shape [B, num_depth_bins, H, W]

        Returns:
            torch.Tensor: Encoded depth feature map of shape [B, 1, H, W]
        """
        objectness = torch.nn.functional.relu(x)
        alpha = torch.clamp(1 - torch.exp(-objectness * self.z_range[2]), 1e-5, 1 - 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat(
                [torch.ones_like(alpha[:, 0:1]), 1-alpha + 1e-10], dim=1
            ), dim=1
        )[:, :-1] # [B, Z, H, W]
        
        depth_map = torch.sum(weights * self.depth_bins[:, None, None], dim=1, keepdim=True)
        return depth_map
    
    

    def _get_scale(self, P2):
        if (self.base_fx is None) or (P2 is None):
            depth_scale = 1
        else:
            input_fx = P2[:, 0, 0] #[B]
            depth_scale = input_fx / self.base_fx #[B]
            depth_scale = depth_scale.reshape([-1, 1, 1, 1]) #[B, 1, 1, 1]
        return depth_scale

    def forward(self, input_features, P2, inputs):
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
                outputs[('logits', i)] = output_logits # [B, Z, H, W]
                outputs[('depth', i, i)] = self._gather_activation(output_logits) * depth_scale # [B, 1, H, W]
        
        B, Z, H, W = outputs[('logits', 0)].shape
        ##
        image_coordinates = project_on_image(
            self.get_coordinates().unsqueeze(0), P2) # [B, X, Y, Z, 3]
        X_index = image_coordinates[..., 0]
        Y_index = image_coordinates[..., 1]
        Z_index = (image_coordinates[..., 2] - self.z_range[0]) / self.z_range[2] # [B, X, Y, Z]
        normed_x_index = X_index / W * 2 - 1 #[B, X, Y, Z]
        normed_y_index = Y_index / H * 2 - 1
        normed_z_index = Z_index / Z * 2 - 1
        _, X3d, Y3d, Z3d = normed_x_index.shape
        rgb_base  = inputs[('original_image', 0)] # [B, 3, H, W]
        RGB = F.grid_sample(
            rgb_base,
            torch.stack([normed_x_index, normed_y_index], dim=-1).reshape(B, 1, X3d*Y3d*Z, 2),
        ).reshape(B, 3, X3d, Y3d, Z3d)

        occupancy = F.grid_sample(
            outputs[('logits', 0)].unsqueeze(1), #[B, 1, Z, H, W]
            torch.stack([normed_x_index, normed_y_index, normed_z_index], dim=-1), #[B, X, Y, Z, 3]
        ) # [B, 1, X, Y, Z]
        # RGB = RGB + self.processing(state)
        outputs['output_voxel'] = torch.cat([RGB, occupancy], dim=1)
        return outputs

    def extract_output_from_pts(self, target_pts, output_dict):
        x = target_pts[..., 0] # [B, 2, Z, pts_batches]
        y = target_pts[..., 1]
        z = target_pts[..., 2]
        x_norm_index = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        y_norm_index = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        z_norm_index = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 2 - 1
        grid_3d_normed = torch.stack([x_norm_index, y_norm_index, z_norm_index], -1) # [B, 2, Z, pts_batches, 3]

        voxel = output_dict['output_voxel'] # [B, 4, X, Y, Z]
        extracted_output = F.grid_sample(voxel, grid_3d_normed, align_corners=True, padding_mode='zeros') # [B, 4, 2, Z, pts_batches]
        return extracted_output

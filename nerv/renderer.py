import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import Unet
from monai.networks.layers import Reshape
from monai.networks.layers.factories import Norm

from monai.transforms import RandRotate90

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet

from pytorch3d.ops.points_to_volumes import _points_to_volumes
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
from pytorch3d.renderer import VolumeRenderer, NDCMultinomialRaysampler, EmissionAbsorptionRaymarcher
from pytorch3d.structures import Pointclouds, Volumes, Meshes

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class NeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
        fwd_renderer=None,
        sh=0,
        pe=0,
        backbone="efficientnet-b7",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.fov_depth = fov_depth
        self.pe = pe
        self.sh = sh
        self.fwd_renderer = fwd_renderer
        assert backbone in backbones.keys()

        self.clarity_net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,  # Condition with straight/hidden view
            out_channels=self.fov_depth,
            num_channels=backbones["efficientnet-b7"],
            attention_levels=[False, False, False, True, True],
            norm_num_groups=16,
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )

    def get_ray_points(self, cameras=None, viewers=None, vol_shape=(1, 256, 256, 256), shape=256):
        B = cameras.R.shape[0]
        _device = cameras.device
        # features = viewers.volumes.features
        # densities = viewers.volumes.density
        # raymarcher = viewers.raymarcher
        raysampler = viewers.raysampler
        renderer = viewers.renderer
        volumes = Volumes(
            features=torch.randn((B, *vol_shape), device=_device),
            densities=torch.randn((B, *vol_shape), device=_device),
            voxel_size=viewers.ndc_extent / shape,
            # volume_translation = [-0.5, -0.5, -0.5],
        )
        
        _, ray_bundle = renderer(cameras=cameras, volumes=volumes) # [...,:3]
        ray_bundle = raysampler.forward(cameras=cameras, n_pts_per_ray=viewers.n_pts_per_ray)
        ray_points = ray_bundle_to_ray_points(ray_bundle).view(B, -1, 3)  
        # ray_points = Pointclouds(ray_points)
        return ray_points

    def forward(self, image2d, cameras, n_views=[2, 1], resample=True, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        if timesteps is None:
            timesteps = torch.zeros((B,), device=_device).long()

        viewpts = torch.cat(
            [
                cameras.R.reshape(B, 1, -1),
                cameras.T.reshape(B, 1, -1),
            ],
            dim=-1,
        )

        clarity = self.clarity_net(
            x=image2d,
            context=viewpts,
            timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        if resample:
            # New methods
            points_3d = self.get_ray_points(cameras=cameras, viewers=self.fwd_renderer)
            points_features = clarity.view(B, -1, 1).float() # B DHW 1
            volume_densities = torch.zeros((B, 1, self.vol_shape, self.vol_shape, self.vol_shape), device=_device)
            volume_features = torch.zeros((B, 1, self.vol_shape, self.vol_shape, self.vol_shape), device=_device)
            grid_sizes = torch.tensor([self.vol_shape, self.vol_shape, self.vol_shape], dtype=torch.int64, device=_device).expand(B, 3)
            point_weight = 1.0
            mask = torch.ones_like(points_features).squeeze(-1)
            align_corners = True
            splat = True
            volume_densities_, volume_features_ = _points_to_volumes(
                points_3d,
                points_features,
                volume_densities,
                volume_features,
                grid_sizes,
                point_weight,
                mask,
                align_corners,
                splat,
            )
            
            values = volume_features_.to(clarity.dtype)
            scenes = torch.split(values, split_size_or_sections=n_views, dim=0)  # 31SHW = [21SHW, 11SHW]
            interp = []
            for scene_, n_view in zip(scenes, n_views):
                value_ = scene_.mean(dim=0, keepdim=True)
                interp.append(value_)

            volumes = torch.cat(interp, dim=0)
        else:
            volumes = F.interpolate(
                density, 
                size=[self.vol_shape, self.vol_shape, self.vol_shape],
                mode="trilinear"
            )
            
        return volumes
        
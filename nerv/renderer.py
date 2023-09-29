import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import Unet
from monai.networks.layers.factories import Norm

from generative.networks.nets import DiffusionModelUNet

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
    "vgg-19": (32, 64, 128, 256, 512),
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
            num_channels=backbones[backbone],
            attention_levels=[False, False, False, True, True],
            norm_num_groups=16,
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )
        
        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=1, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.INSTANCE, 
                dropout=0.5
            )
        )
        
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

        density = self.density_net(clarity)
        return density, clarity
        
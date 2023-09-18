import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    look_at_rotation,
    look_at_view_transform,
)

from monai.networks.nets import Unet
from monai.networks.layers import Reshape
from monai.networks.layers.factories import Norm

from generative.inferers import DiffusionInferer
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
}



class NeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
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
        
        # self.density_net = DiffusionModelUNet(
        #     spatial_dims=3,
        #     in_channels=1,  # Condition with straight/hidden view
        #     out_channels=1,
        #     num_channels=backbones["efficientnet-b1"],
        #     attention_levels=[False, False, False, False, True],
        #     norm_num_groups=8,
        #     num_res_blocks=2,
        #     with_conditioning=True,
        #     cross_attention_dim=12,  # flatR | flatT
        # )
        
        # self.density_net = nn.Sequential(
        #     Unet(
        #         spatial_dims=3, 
        #         in_channels=1 + (2 * 3 * self.pe), 
        #         out_channels=1, 
        #         channels=backbones[backbone], 
        #         strides=(2, 2, 2, 2, 2), 
        #         num_res_units=2, 
        #         kernel_size=3, 
        #         up_kernel_size=3, 
        #         act=("LeakyReLU", {"inplace": True}), 
        #         norm=Norm.BATCH, 
        #         dropout=0.5,
        #     ),
        #     # nn.Tanh(),
        # )

        # self.mixture_net = nn.Sequential(
        #     Unet(
        #         spatial_dims=3, 
        #         in_channels=2 + (2 * 3 * self.pe), 
        #         out_channels=1, 
        #         channels=backbones[backbone], 
        #         strides=(2, 2, 2, 2, 2), 
        #         num_res_units=2, 
        #         kernel_size=3, 
        #         up_kernel_size=3, 
        #         act=("LeakyReLU", {"inplace": True}), 
        #         norm=Norm.BATCH, 
        #         dropout=0.5,
        #     ),
        #     # nn.Tanh(),
        # )

        # self.refiner_net = nn.Sequential(
        #     Unet(
        #         spatial_dims=3, 
        #         in_channels=3 + (2 * 3 * self.pe), 
        #         out_channels=self.out_channels, 
        #         channels=backbones[backbone], 
        #         strides=(2, 2, 2, 2, 2), 
        #         num_res_units=2, 
        #         kernel_size=3, 
        #         up_kernel_size=3, 
        #         act=("LeakyReLU", {"inplace": True}), 
        #         norm=Norm.BATCH, 
        #         dropout=0.5
        #     ),
        #     # nn.Tanh(),
        # )

    def forward(self, image2d, cameras, n_views=[2, 1], resample=True, timesteps=None):
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
            z = torch.linspace(-1.0, 1.0, steps=self.vol_shape, device=_device)
            y = torch.linspace(-1.0, 1.0, steps=self.vol_shape, device=_device)
            x = torch.linspace(-1.0, 1.0, steps=self.vol_shape, device=_device)
            coords = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(B, 1, 1)  # 1 DHW 3 to B DHW 3
            
            # Process (resample) the volumes from ray views to ndc
            points = cameras.transform_points_ndc(coords)  # world to ndc, 1 DHW 3
            values = F.grid_sample(
                clarity, 
                points.view(-1, self.vol_shape, self.vol_shape, self.vol_shape, 3), 
                mode="bilinear", 
                padding_mode="border", 
                align_corners=False,
            )

            scenes = torch.split(values, split_size_or_sections=n_views, dim=0)  # 31SHW = [21SHW, 11SHW]
            interp = []
            for scene_, n_view in zip(scenes, n_views):
                value_ = scene_.mean(dim=0, keepdim=True)
                interp.append(value_)

            clarity = torch.cat(interp, dim=0)
        else:
            clarity = F.interpolate(
                clarity, 
                size=[self.vol_shape, self.vol_shape, self.vol_shape],
                mode="trilinear"
            )
            
        return clarity
        # shcomps = self.density_net(
        #     x=clarity,
        #     context=viewpts,
        #     timesteps=timesteps,
        # ) 
        
        # density = self.density_net(torch.cat([clarity], dim=1))  # density = torch.add(density, clarity)
        # mixture = self.mixture_net(torch.cat([clarity, density], dim=1))  # mixture = torch.add(mixture, clarity)
        # shcoeff = self.refiner_net(torch.cat([clarity, density, mixture], dim=1))  # shcoeff = torch.add(shcoeff, clarity)
        # shcomps = shcoeff

        # volumes = []
        # for idx, n_view in enumerate(n_views):
        #     volume = shcomps[[idx]].repeat(n_view, 1, 1, 1, 1)
        #     volumes.append(volume)
        # volumes = torch.cat(volumes, dim=0)
        
        # return volumes

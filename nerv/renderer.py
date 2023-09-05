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
    def __init__(self, in_channels=1, out_channels=1, img_shape=256, vol_shape=256, n_pts_per_ray=400, sh=0, pe=8, backbone="efficientnet-b7") -> None:
        super().__init__()
        self.sh = sh
        self.pe = pe
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.n_pts_per_ray = n_pts_per_ray
        assert backbone in backbones.keys()
        if self.pe > 0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1)  # torch.Size([100, 100, 100, 3])
            num_frequencies = self.pe
            min_freq_exp = 0
            max_freq_exp = 8
            encoder = encodings.NeRFEncoding(in_dim=self.pe, num_frequencies=num_frequencies, min_freq_exp=min_freq_exp, max_freq_exp=max_freq_exp,)
            pebasis = encoder(zyx.view(-1, 3))
            pebasis = pebasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer("pebasis", pebasis)

        if self.sh > 0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1)  # torch.Size([100, 100, 100, 3])

            encoder = encodings.SHEncoding(self.sh)
            assert out_channels == self.sh ** 2 if self.sh > 0 else 1
            shbasis = encoder(zyx.view(-1, 3))
            shbasis = shbasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer("shbasis", shbasis)

        self.clarity_net = nn.Sequential(
            Unet(spatial_dims=2, 
                in_channels=1, 
                out_channels=self.vol_shape, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH, 
                dropout=0.5,
            )
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=1 + (2 * 3 * self.pe), 
                out_channels=1, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH, 
                dropout=0.5,
            )
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=2 + (2 * 3 * self.pe), 
                out_channels=1, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH, 
                dropout=0.5
            )
        )

        self.refiner_net = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=3 + (2 * 3 * self.pe), 
                out_channels=self.out_channels, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH, 
                dropout=0.5
            )
        )

    def forward(
        self, image2d, cameras, n_views=[2, 1], resample=True,
    ):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        clarity = self.clarity_net(image2d).view(-1, 1, self.vol_shape, self.img_shape, self.img_shape)

        density = self.density_net(torch.cat([clarity], dim=1))  # density = torch.add(density, clarity)
        mixture = self.mixture_net(torch.cat([clarity, density], dim=1))  # mixture = torch.add(mixture, clarity)
        shcoeff = self.refiner_net(torch.cat([clarity, density, mixture], dim=1))  # shcoeff = torch.add(shcoeff, clarity)
        shcomps = shcoeff

        volumes = []
        for idx, n_view in enumerate(n_views):
            volume = shcomps[[idx]].repeat(n_view, 1, 1, 1, 1)
            volumes.append(volume)
        volumes = torch.cat(volumes, dim=0)

        if resample:
            pass

        return volumes

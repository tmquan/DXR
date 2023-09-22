import os
import warnings

warnings.filterwarnings("ignore")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_float32_matmul_precision("high")

from typing import Optional, NamedTuple
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import MultiScalePatchDiscriminator

from argparse import ArgumentParser

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
from nerv.renderer import NeRVFrontToBackInverseRenderer

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


def make_cameras_dea(dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor, fov: int = 10, znear: int = 18.0, zfar: int = 22.0, is_orthogonal: bool = False):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist.float(), elev=elev.float() * 90, azim=azim.float() * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)


class DXRLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr

        self.ckpt = hparams.ckpt
        self.gan = hparams.gan
        self.strict = hparams.strict
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.fov_depth = hparams.fov_depth
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lamda = hparams.lamda
        self.timesteps = hparams.timesteps

        self.logsdir = hparams.logsdir
        self.sh = hparams.sh
        self.pe = hparams.pe

        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.backbone = hparams.backbone

        self.save_hyperparameters()

        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=4.0, 
            max_depth=8.0, 
            ndc_extent=2.0,
        )

        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh ** 2 if self.sh > 0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            fov_depth=self.fov_depth, 
            sh=self.sh, 
            pe=self.pe, 
            backbone=self.backbone, 
            fwd_renderer=self.fwd_renderer,
        )

        if self.ckpt:
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.l1loss = nn.L1Loss(reduction="mean")
        
        if self.gan:
            self.automatic_optimization = False
            self.discriminator = MultiScalePatchDiscriminator(
                num_d=2,
                num_layers_d=3,
                spatial_dims=2,
                num_channels=32,
                in_channels=1,
                out_channels=8,
                activation="LEAKYRELU",
                minimum_size_im=self.img_shape,
                norm="INSTANCE",
                kernel_size=3, 
                dropout=0.2
            )
            self.adv_loss = PatchAdversarialLoss(reduction="sum", criterion="hinge")
            # self.p2d_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            self.p2d_loss = PerceptualLoss(
                spatial_dims=2, 
                network_type="radimagenet_resnet50", 
                is_fake_3d=False, 
                pretrained=True
            )
            self.p3d_loss = PerceptualLoss(
                spatial_dims=3, 
                network_type="radimagenet_resnet50", 
                is_fake_3d=True, fake_3d_ratio=0.1,
                pretrained=True
            )
            
        self.psnr = PeakSignalNoiseRatio(data_range=(-1, 1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1, 1))
        self.psnr_outputs = []
        self.ssim_outputs = []
    
    def feat_loss(self, input_features_disc_fake, input_features_disc_real, lambda_feat=0.1):
        num_D = len(input_features_disc_fake)
        GAN_Feat_loss = torch.zeros(1).to(input_features_disc_fake[0][0].device)
        for i in range(num_D):  # for each discriminator
            num_intermediate_outputs = len(input_features_disc_fake[i])
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.l1loss(input_features_disc_fake[i][j], input_features_disc_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * lambda_feat / num_D
        return GAN_Feat_loss

    def forward_screen(self, image3d, cameras):
        screen = self.fwd_renderer(image3d * 0.5 + 0.5 / image3d.shape[1], cameras) 
        screen = screen * 2.0 - 1.0
        return screen

    def forward_volume(self, image2d, cameras, n_views=[2, 1], resample=True, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        results = self.inv_renderer(image2d, cameras, n_views, resample)
        # results, middles = self.inv_renderer(image2d, cameras, n_views, resample)
        #.view(-1, 1, self.vol_shape, self.vol_shape, self.vol_shape)
        # if is_training:
        #     return results, middles
        return results
        

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = "evaluation"):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]
        
        optimizer_g, optimizer_d = self.optimizers()
        
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=18, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=18, znear=4, zfar=8)
        
        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        
        # Reconstruct the Encoder-Decoder
        # volume_dx_inverse, \
        # middle_dx_inverse = self.forward_volume(
        #     image2d=torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]), 
        #     cameras=join_cameras_as_batch([view_hidden, view_random, view_hidden]), 
        #     n_views=[1, 1, 1] * batchsz,
        #     is_training=True)
        volume_dx_inverse = self.forward_volume(
            image2d=torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]), 
            cameras=join_cameras_as_batch([view_hidden, view_random, view_hidden]), 
            n_views=[1, 1, 1] * batchsz,
            is_training=True)
        (volume_xr_hidden_inverse, volume_ct_random_inverse, volume_ct_hidden_inverse,) = torch.split(volume_dx_inverse, batchsz)
        # (middle_xr_hidden_inverse, middle_ct_random_inverse, middle_ct_hidden_inverse,) = torch.split(middle_dx_inverse, batchsz)
        # volume_dx_inverse = self.forward_volume(
        #     image2d=torch.cat([figure_xr_hidden, figure_ct_hidden]), 
        #     cameras=join_cameras_as_batch([view_hidden, view_hidden]), 
        #     n_views=[1, 1] * batchsz,)
        # (volume_xr_hidden_inverse, volume_ct_hidden_inverse,) = torch.split(volume_dx_inverse, batchsz)

        figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)

        if self.sh > 0:
            volume_xr_hidden_inverse = volume_xr_hidden_inverse.sum(dim=1, keepdim=True)
            volume_ct_random_inverse = volume_ct_random_inverse.sum(dim=1, keepdim=True)
            volume_ct_hidden_inverse = volume_ct_hidden_inverse.sum(dim=1, keepdim=True)

        im2d_loss_inv = (
            self.l1loss(figure_xr_hidden_inverse_hidden, figure_xr_hidden)
            + self.l1loss(figure_ct_random_inverse_random, figure_ct_random)
            + self.l1loss(figure_ct_random_inverse_hidden, figure_ct_hidden)
            + self.l1loss(figure_ct_hidden_inverse_random, figure_ct_random) * self.omega
            + self.l1loss(figure_ct_hidden_inverse_hidden, figure_ct_hidden) * self.omega
        )

        # im3d_loss_inv = self.l1loss(volume_ct_hidden_inverse, image3d) \
        #               + self.l1loss(volume_ct_random_inverse, image3d) \
        #               + self.l1loss(middle_ct_hidden_inverse, image3d) \
        #               + self.l1loss(middle_ct_random_inverse, image3d)
        
        im3d_loss_inv = self.l1loss(volume_ct_hidden_inverse, image3d) \
                      + self.l1loss(volume_ct_random_inverse, image3d)

        im2d_loss = im2d_loss_inv
        im3d_loss = im3d_loss_inv

        # Log the final losses
        self.log(f"train_im2d_loss", im2d_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        self.log(f"train_im3d_loss", im3d_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)

        # Visualization step
        if batch_idx == 0:
            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    image2d, 
                    volume_xr_hidden_inverse[..., self.vol_shape // 2, :], 
                    figure_xr_hidden_inverse_random, 
                    figure_xr_hidden_inverse_hidden, 
                    image3d[..., self.vol_shape // 2, :], 
                    figure_ct_random, 
                    figure_ct_hidden,
                ], dim=-2,).transpose(2, 3),
                torch.cat([
                    zeros,
                    # zeros,
                    # zeros,
                    # zeros,
                    # volume_xr_sample_hidden[..., self.vol_shape // 2, :],
                    # figure_xr_sample_hidden_random,
                    # figure_xr_sample_hidden_hidden,
                    volume_ct_random_inverse[..., self.vol_shape // 2, :],
                    figure_ct_random_inverse_random,
                    figure_ct_random_inverse_hidden,
                    volume_ct_hidden_inverse[..., self.vol_shape // 2, :],
                    figure_ct_hidden_inverse_random,
                    figure_ct_hidden_inverse_hidden,
                ], dim=-2,).transpose(2, 3),
            ], dim=-2,)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(-1.0, 1.0) * 0.5 + 0.5
            tensorboard.add_image(f"train_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx,)
        
        
        loss = self.alpha * im3d_loss + self.gamma * im2d_loss

        # adversarial loss is binary cross-entropy
        disc_fakes, features_fakes = self.discriminator(figure_xr_hidden_inverse_random)
        disc_reals, features_reals = self.discriminator(figure_ct_random)
        
        loss_g = self.adv_loss(disc_fakes, target_is_real=True, for_discriminator=False)
        loss_feat = self.feat_loss(features_fakes, features_reals, 1)
        loss_perc = self.p2d_loss(figure_xr_hidden_inverse_random*0.5+0.5, figure_ct_random*0.5+0.5) \
                  + self.p3d_loss(volume_xr_hidden_inverse*0.5+0.5, image3d*0.5+0.5) \

        self.log("loss_g", loss_g, prog_bar=True)
        self.manual_backward(loss_g + loss_feat + loss_perc * self.lamda + loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        disc_fakes, _ = self.discriminator(figure_xr_hidden_inverse_random.detach())
        loss_d_r = self.adv_loss(disc_reals, target_is_real=True, for_discriminator=True)
        loss_g_f = self.adv_loss(disc_fakes, target_is_real=False, for_discriminator=True)
        optimizer_d.zero_grad()
        loss_d = loss_d_r + loss_g_f

        self.log("loss_d", loss_d, prog_bar=True)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]
        
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=18, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=18, znear=4, zfar=8)
        
        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        # Reconstruct the Encoder-Decoder
        volume_dx_inverse = self.forward_volume(
            image2d=torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]), 
            cameras=join_cameras_as_batch([view_hidden, view_random, view_hidden]), 
            n_views=[1, 1, 1] * batchsz,)
        (volume_xr_hidden_inverse, volume_ct_random_inverse, volume_ct_hidden_inverse,) = torch.split(volume_dx_inverse, batchsz)
        # volume_dx_inverse = self.forward_volume(
        #     image2d=torch.cat([figure_xr_hidden, figure_ct_hidden]), 
        #     cameras=join_cameras_as_batch([view_hidden, view_hidden]), 
        #     n_views=[1, 1] * batchsz,)
        # (volume_xr_hidden_inverse, volume_ct_hidden_inverse,) = torch.split(volume_dx_inverse, batchsz)

        figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)

        if self.sh > 0:
            volume_xr_hidden_inverse = volume_xr_hidden_inverse.sum(dim=1, keepdim=True)
            volume_ct_random_inverse = volume_ct_random_inverse.sum(dim=1, keepdim=True)
            volume_ct_hidden_inverse = volume_ct_hidden_inverse.sum(dim=1, keepdim=True)

        im2d_loss_inv = (
            self.l1loss(figure_xr_hidden_inverse_hidden, figure_xr_hidden)
            + self.l1loss(figure_ct_random_inverse_random, figure_ct_random)
            + self.l1loss(figure_ct_random_inverse_hidden, figure_ct_hidden)
            + self.l1loss(figure_ct_hidden_inverse_random, figure_ct_random) * self.omega
            + self.l1loss(figure_ct_hidden_inverse_hidden, figure_ct_hidden) * self.omega
        )

        im3d_loss_inv = self.l1loss(volume_ct_hidden_inverse, image3d) + self.l1loss(volume_ct_random_inverse, image3d)

        im2d_loss = im2d_loss_inv
        im3d_loss = im3d_loss_inv

        # Log the final losses
        self.log(f"validation_im2d_loss", im2d_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        self.log(f"validation_im3d_loss", im3d_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)

        # Visualization step
        if batch_idx == 0:
            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    image2d, 
                    volume_xr_hidden_inverse[..., self.vol_shape // 2, :], 
                    figure_xr_hidden_inverse_random, 
                    figure_xr_hidden_inverse_hidden, 
                    image3d[..., self.vol_shape // 2, :], 
                    figure_ct_random, 
                    figure_ct_hidden,
                ], dim=-2,).transpose(2, 3),
                torch.cat([
                    zeros,
                    # zeros,
                    # zeros,
                    # zeros,
                    # volume_xr_sample_hidden[..., self.vol_shape // 2, :],
                    # figure_xr_sample_hidden_random,
                    # figure_xr_sample_hidden_hidden,
                    volume_ct_random_inverse[..., self.vol_shape // 2, :],
                    figure_ct_random_inverse_random,
                    figure_ct_random_inverse_hidden,
                    volume_ct_hidden_inverse[..., self.vol_shape // 2, :],
                    figure_ct_hidden_inverse_random,
                    figure_ct_hidden_inverse_hidden,
                ], dim=-2,).transpose(2, 3),
            ], dim=-2,)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(-1.0, 1.0) * 0.5 + 0.5
            tensorboard.add_image(f"validation_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx,)
        
        
        loss = self.alpha * im3d_loss + self.gamma * im2d_loss
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=18, znear=4, zfar=8)

        # Construct the samples in 2D
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        figure_xr_hidden = image2d

        # Reconstruct the Encoder-Decoder
        volume_ct_hidden = self.forward_volume(image2d=figure_ct_hidden)
        psnr = self.psnr(volume_ct_hidden, image3d)
        ssim = self.ssim(volume_ct_hidden, image3d)
        self.psnr_outputs.append(psnr)
        self.ssim_outputs.append(ssim)
        
    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f"train_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True,)
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f"validation_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True,)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        print(f"PSNR :{torch.stack(self.psnr_outputs).mean()}")
        print(f"SSIM :{torch.stack(self.ssim_outputs).mean()}")
        self.psnr_outputs.clear()
        self.ssim_outputs.clear()
        
    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            [
                {"params": self.inv_renderer.parameters()},
                # {'params': self.unet2d_model.parameters()}, # Add diffusion model, remove gan model
            ],
            lr=self.lr,
            betas=(0.5, 0.999)
            # self.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 200], gamma=0.1)
        
        optimizer_d = torch.optim.AdamW(
            [
                {"params": self.discriminator.parameters()},
            ],
            lr=self.lr,
            betas=(0.5, 0.999)
            # self.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[100, 200], gamma=0.1)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--devices", default=None)

    # Model arguments
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--fov_depth", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--test", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--gan", action="store_true", help="train with gan xray ct random")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")

    parser.add_argument("--alpha", type=float, default=1.0, help="vol loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="img loss")
    parser.add_argument("--delta", type=float, default=1.0, help="vgg loss")
    parser.add_argument("--theta", type=float, default=1.0, help="cam loss")
    parser.add_argument("--omega", type=float, default=1.0, help="cam cond")
    parser.add_argument("--lamda", type=float, default=0.1, help="perc loss")

    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_true", help="training strategy")
    parser.add_argument("--backbone", type=str, default="efficientnet-b7", help="Backbone for network")
    parser.add_argument("--prediction_type", type=str, default="sample", help="prediction_type for network",)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}",
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=f"{hparams.logsdir}", log_graph=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [
        lr_callback,
        checkpoint_callback,
    ]
    # if hparams.strategy != "gan":
    #     callbacks.append(swa_callback)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=callbacks,
        # accumulate_grad_batches=4,
        strategy=hparams.strategy,  # "auto", #"ddp_find_unused_parameters_true",
        precision=16 if hparams.amp else 32,
        profiler="advanced",
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4",),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
    ]

    train_label3d_folders = []

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = []

    val_image3d_folders = [
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4",),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2020/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2021/images'),
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DXRLightningModule(hparams=hparams)
    
    if hparams.test:
        trainer.test(
            model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=hparams.ckpt
        )

    else:
        trainer.fit(
            model,
            # compiled_model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            # datamodule=datamodule,
            ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None,  # "some/path/to/my_checkpoint.ckpt"
        )

    # serve

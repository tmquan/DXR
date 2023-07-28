import os
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torch.set_float32_matmul_precision('high')

from typing import Optional, NamedTuple
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from argparse import ArgumentParser
from tqdm.auto import tqdm

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras, 
    look_at_view_transform, 
)

from monai.networks.nets import Unet
from monai.networks.layers import Reshape
from monai.networks.layers.factories import Norm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
from nerv.renderer import NeRVFrontToBackInverseRenderer, backbones

def make_cameras_dea(
    dist: torch.Tensor, 
    elev: torch.Tensor, 
    azim: torch.Tensor, 
    fov: int=10, 
    znear: int=18.0, 
    zfar: int=22.0
    ):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(
        dist=dist.float(), 
        elev=elev.float() * 90, 
        azim=azim.float() * 180
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)

class DXRLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        
        self.ckpt = hparams.ckpt
        self.lpips = hparams.lpips
        self.strict = hparams.strict
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
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
            ndc_extent=4.0,
        )
        
        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh**2 if self.sh>0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray,
            sh=self.sh, 
            pe=self.pe,
            backbone=self.backbone,
        )
        
        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1, # Condition with straight/hidden view
            out_channels=1,
            num_channels=backbones[self.backbone],
            attention_levels=[False, False, False, True, True],
            norm_num_groups=16, 
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=12, #flatR | flatT
        )
        self.ddpmsch = DDPMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type,
            beta_start=0.0005, beta_end=0.0195)
        self.ddimsch = DDIMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type,
            beta_start=0.0005, beta_end=0.0195, clip_sample=True)
        self.ddimsch.set_timesteps(num_inference_steps=100)
        self.inferer = DiffusionInferer(scheduler=self.ddimsch)
        
        if self.ckpt:
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)
 

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.l1loss = nn.L1Loss(reduction="mean")
        self.lpips_ = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        
    def forward_screen(self, image3d, cameras):   
        return self.fwd_renderer(image3d * 0.5 + 0.5/image3d.shape[1], cameras) * 2.0 - 1.0

    def forward_volume(self, image2d, cameras, n_views=[2, 1], resample_clarity=True, resample_volumes=False): 
        return self.inv_renderer(image2d, 
                                 cameras, 
                                 n_views, 
                                 resample_clarity=resample_clarity, 
                                 resample_volumes=resample_volumes)
        
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]
        
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1 # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=30, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=30, znear=4, zfar=8)
        
        # Construct the samples in 2D
        figure_xr_hidden = image2d 
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        
        # Diffusion timesteps
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batchsz,), device=_device).long()
        
        # Construct the context pose to diffusion model
        pose_random = torch.cat([view_random.R.reshape(batchsz, 1, -1), view_random.T.reshape(batchsz, 1, -1)], dim=-1)
        pose_hidden = torch.cat([view_hidden.R.reshape(batchsz, 1, -1), view_hidden.T.reshape(batchsz, 1, -1)], dim=-1)    
     
        # Reconstruct the Encoder-Decoder
        volume_dx_inverse = self.forward_volume(
            image2d=torch.cat([figure_xr_hidden, 
                                figure_ct_random, 
                                figure_ct_hidden]),
            cameras=join_cameras_as_batch([view_hidden, view_random, view_hidden]),
            n_views=[1, 1, 1])
        volume_xr_hidden_inverse,\
        volume_ct_random_inverse,\
        volume_ct_hidden_inverse = torch.split(volume_dx_inverse, batchsz)
        
        figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)
        
        if self.sh>0:
            volume_xr_hidden_inverse = volume_xr_hidden_inverse.sum(dim=1, keepdim=True)
            volume_ct_random_inverse = volume_ct_random_inverse.sum(dim=1, keepdim=True)
            volume_ct_hidden_inverse = volume_ct_hidden_inverse.sum(dim=1, keepdim=True)
        
        im2d_loss_inv = self.l1loss(figure_xr_hidden_inverse_hidden, figure_xr_hidden) \
                      + self.l1loss(figure_ct_random_inverse_random, figure_ct_random) \
                      + self.l1loss(figure_ct_random_inverse_hidden, figure_ct_hidden) \
                      + self.l1loss(figure_ct_hidden_inverse_random, figure_ct_random) \
                      + self.l1loss(figure_ct_hidden_inverse_hidden, figure_ct_hidden) 
                        
        if self.lpips:
            figure_xr_hidden_inverse_random = torch.nan_to_num(figure_xr_hidden_inverse_random, 0, 1, -1)
            lpips_loss = self.lpips_(figure_xr_hidden_inverse_random.repeat(1, 3, 1, 1).clamp(-1, 1), 
                                     figure_ct_random.repeat(1, 3, 1, 1).clamp(-1, 1)) 
            self.log(f'{stage}_lpip_loss', lpips_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            im2d_loss_inv += lpips_loss
                    
        im3d_loss_inv = self.l1loss(volume_ct_random_inverse, image3d) \
                      + self.l1loss(volume_ct_hidden_inverse, image3d)  
        
        # Diffusion step: 2 kinds of blending
        volume_xr_latent = torch.randn_like(image3d)
        figure_xr_latent_hidden = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
        # figure_xr_latent_hidden = torch.randn_like(image2d)
        figure_xr_interp_hidden = self.ddpmsch.add_noise(
            original_samples=image2d, 
            noise=figure_xr_latent_hidden, 
            timesteps=timesteps
        )
        
        volume_ct_latent = torch.randn_like(image3d)
        figure_ct_latent_random = self.forward_screen(image3d=volume_ct_latent, cameras=view_random)
        figure_ct_latent_hidden = self.forward_screen(image3d=volume_ct_latent, cameras=view_hidden)
        
        volume_ct_interp = self.ddpmsch.add_noise(
            original_samples=image3d, 
            noise=volume_ct_latent, 
            timesteps=timesteps
        )
        figure_ct_interp_random = self.forward_screen(image3d=volume_ct_interp, cameras=view_random)
        figure_ct_interp_hidden = self.forward_screen(image3d=volume_ct_interp, cameras=view_hidden)
        
        # Run the backward diffusion (denoising + reproject)
        figure_dx_interp = torch.cat([
            figure_xr_interp_hidden,
            figure_ct_interp_random, 
            figure_ct_interp_hidden, 
        ])
        pose_dx_interp = torch.cat([
            pose_hidden,
            pose_random, 
            pose_hidden, 
        ])
        figure_dx_output = self.unet2d_model(
            x=figure_dx_interp, 
            context=pose_dx_interp,
            timesteps=timesteps
        )
        
        figure_xr_output_hidden,\
        figure_ct_output_random,\
        figure_ct_output_hidden = torch.split(figure_dx_output, batchsz)
        
        # # Reconstruct the Encoder-Decoder
        # volume_ct_random_output = self.forward_volume(
        #     image2d=figure_ct_output_random,
        #     cameras=view_random,
        #     n_views=[1])
        
        # if self.sh>0:
        #     volume_ct_random_output = volume_ct_random_output.sum(dim=1, keepdim=True)
            
        if self.ddpmsch.prediction_type == "sample":
            figure_xr_target_hidden = figure_xr_hidden
            figure_ct_target_random = figure_ct_random
            figure_ct_target_hidden = figure_ct_hidden
            volume_ct_target = image3d
        elif self.ddpmsch.prediction_type == "epsilon":
            figure_xr_target_hidden = figure_xr_latent_hidden
            figure_ct_target_random = figure_ct_latent_random
            figure_ct_target_hidden = figure_ct_latent_hidden
            volume_ct_target = volume_ct_latent
        elif self.ddpmsch.prediction_type == "v_prediction":
            figure_xr_target_hidden = self.ddpmsch.get_velocity(figure_xr_hidden, figure_xr_latent_hidden, timesteps)
            figure_ct_target_random = self.ddpmsch.get_velocity(figure_ct_random, figure_ct_latent_random, timesteps)
            figure_ct_target_hidden = self.ddpmsch.get_velocity(figure_ct_hidden, figure_ct_latent_hidden, timesteps)
            volume_ct_target = self.ddpmsch.get_velocity(image3d, volume_ct_latent, timesteps)
    
        # im3d_loss_dif = self.l1loss(volume_ct_random_output, volume_ct_target) 
        
        im2d_loss_dif = self.l1loss(figure_xr_output_hidden, figure_xr_target_hidden) \
                      + self.l1loss(figure_ct_output_random, figure_ct_target_random) \
                      + self.l1loss(figure_ct_output_hidden, figure_ct_target_hidden) 

        # im2d_loss = im2d_loss_inv + im2d_loss_dif
        # im3d_loss = im3d_loss_inv + im3d_loss_dif
            
        im2d_loss = im2d_loss_inv + im2d_loss_dif
        im3d_loss = im3d_loss_inv
                                
        # Log the final losses
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
                         
        # Visualization step 
        if batch_idx==0:
            # Sampling step for X-ray
            with torch.no_grad():
                volume_xr_latent = torch.randn_like(image3d)
                figure_xr_latent_random = self.forward_screen(image3d=volume_xr_latent, cameras=view_random)
                figure_xr_latent_hidden = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
                
                figure_xr_sample_random = self.inferer.sample(
                    input_noise=figure_xr_latent_random, 
                    diffusion_model=self.unet2d_model, 
                    conditioning=pose_random,
                    scheduler=self.ddimsch, 
                    verbose=False
                )
                
                figure_xr_sample_hidden = self.inferer.sample(
                    input_noise=figure_xr_latent_hidden, 
                    diffusion_model=self.unet2d_model, 
                    conditioning=pose_hidden,
                    scheduler=self.ddimsch, 
                    verbose=False
                )
                                
                volume_xr_sample = self.forward_volume(
                    image2d=torch.cat([figure_xr_sample_random, figure_xr_sample_hidden]),
                    cameras=join_cameras_as_batch([view_random, view_hidden]),
                    n_views=[1, 1]
                )
                
                volume_xr_sample_random,\
                volume_xr_sample_hidden = torch.split(volume_xr_sample, batchsz)
                figure_xr_sample_random_random = self.forward_screen(image3d=volume_xr_sample_random, cameras=view_random)
                figure_xr_sample_random_hidden = self.forward_screen(image3d=volume_xr_sample_random, cameras=view_hidden)
                figure_xr_sample_hidden_random = self.forward_screen(image3d=volume_xr_sample_hidden, cameras=view_random)
                figure_xr_sample_hidden_hidden = self.forward_screen(image3d=volume_xr_sample_hidden, cameras=view_hidden)
                    
            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    image2d, 
                    volume_xr_hidden_inverse[..., self.vol_shape//2, :],                   
                    figure_xr_hidden_inverse_random, 
                    figure_xr_hidden_inverse_hidden, 
                    image3d[..., self.vol_shape//2, :],
                    figure_ct_random, 
                    figure_ct_hidden, 
                ], dim=-2).transpose(2, 3),     
                torch.cat([
                    zeros,
                    volume_ct_random_inverse[..., self.vol_shape//2, :], 
                    figure_ct_random_inverse_random, 
                    figure_ct_random_inverse_hidden,
                    volume_ct_hidden_inverse[..., self.vol_shape//2, :], 
                    figure_ct_hidden_inverse_random, 
                    figure_ct_hidden_inverse_hidden,
                ], dim=-2).transpose(2, 3),     
                torch.cat([
                    zeros,
                    volume_xr_sample_random[..., self.vol_shape//2, :], 
                    figure_xr_sample_random_random, 
                    figure_xr_sample_random_hidden,
                    volume_xr_sample_hidden[..., self.vol_shape//2, :], 
                    figure_xr_sample_hidden_random, 
                    figure_xr_sample_hidden_hidden,
                ], dim=-2).transpose(2, 3),                    
            ], dim=-2)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(-1., 1.) * 0.5 + 0.5 
            tensorboard.add_image(f'{stage}_df_samples', grid2d, self.current_epoch*self.batch_size + batch_idx)
                
        loss = self.alpha*im3d_loss + self.gamma*im2d_loss
        return loss
            
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._common_step(batch, batch_idx, optimizer_idx, stage='train')
        self.train_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, optimizer_idx=-1, stage='validation')
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f'train_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory
        
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f'validation_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # [
            #     {'params': self.inv_renderer.parameters()},
            #     {'params': self.unet2d_model.parameters()}, # Add diffusion model, remove lpips model
            # ], lr=self.lr, betas=(0.9, 0.999)
            self.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]
    
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
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--lpips", action="store_true", help="train with lpips xray ct random")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--delta", type=float, default=1., help="vgg loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--strategy", type=str, default='auto', help="training strategy")
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--prediction_type", type=str, default='sample', help="prediction_type for network")
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
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{hparams.logsdir}", 
        log_graph=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks=[
        lr_callback,
        checkpoint_callback,
    ]
    if hparams.strategy!= "fsdp":
         callbacks.append(swa_callback)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=callbacks,
        accumulate_grad_batches=4,
        strategy=hparams.strategy, #"auto", #"ddp_find_unused_parameters_true", 
        precision=16 if hparams.amp else 32,
        profiler="advanced"
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = [
    ]

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
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
        vol_shape=hparams.vol_shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DXRLightningModule(
        hparams=hparams
    )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    trainer.fit(
        model,
        # compiled_model,
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=datamodule.val_dataloader(),
        # datamodule=datamodule,
        ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve
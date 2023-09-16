import os
import torch
import nibabel as nib
import numpy as np
import tempfile
import monai

from tqdm.auto import tqdm
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
from monai.config import print_config

from argparse import ArgumentParser
# from your_model_module import YourLightningModel  # Import your Lightning model class
from datamodule import UnpairedDataModule  # Import your data module class

def main(ckpt, datamodule, batch_size):
    pass
    # # Initialize the data module and load data
    # datamodule = YourDataModule(data_dir, batch_size=batch_size)
    # datamodule.setup()

    # # Load the model checkpoint
    # model = YourLightningModel.load_from_checkpoint(checkpoint_path)

    # # Put the model in evaluation mode (important for models with dropout, batch normalization, etc.)
    # model.eval()

    # # Inference loop (modify this according to your needs)
    # predictions = []
    # for batch in datamodule.test_dataloader():
    #     inputs = batch  # Modify this if your model requires preprocessing of inputs
    #     with torch.no_grad():
    #         outputs = model(*inputs)  # Modify this to match your forward pass
    #     predictions.append(outputs)  # Modify this to store your model's outputs

    # # Post-process the predictions if needed

    # # Print or save the predictions
    # print(predictions)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--devices", default=None)

    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--lpips", action="store_true", help="train with lpips xray ct random")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")

    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()
    
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
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = []

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
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
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2020/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2021/images'),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
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
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
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
        vol_shape=hparams.vol_shape,
    )
    datamodule.setup()
    print(len(datamodule.val_dataloader()))
    
    # using the Nibabel backend
    saver3d = SaveImage(
        output_dir="../results/groundtruth",
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        # squeeze_end_dims=False,
        # writer="NibabelWriter",
    )
    # saver3d.set_options(data_kwargs={"spatial_ndim": 3})
    
    for batch in tqdm(datamodule.val_dataloader()):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        image3d_meta_dict = batch["image3d_meta_dict"]
        image2d_meta_dict = batch["image2d_meta_dict"]
        _device = batch["image3d"].device
        
        # Save the groundtruth
        image3d = saver3d(image3d[0])
import glob, os
from random import *
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import time, math

import torchio as tio

from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss, FocalLoss
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.networks.nets import UNet, DynUNet
from monai.transforms import (
    AsDiscrete, Compose, LoadNiftid, ToTensord, AddChanneld, LabelToContour, Spacingd
)
from monai.utils import set_determinism
from network.deeplabv3_3d import DeepLabV3_3D
from Ki_UNet_remake import kiunet3dwcrfb_o
from network.UNet_3Plus_3D_drop import UNet_3Plus_3D, UNet_3Plus_DeepSup_3D

os.environ["MONAI_DATA_DIRECTORY"] = "./data"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = directory
print(root_dir)

### hyperparameter setting
set_determinism(seed=0)

bs = 4
Height = 96
Width = Height
Depth = 16
epoch_num = 200
l_rate = 5e-5
network = "kiunet"
precision = 32
gpu_num = 4
dropout = 0.1
model_channels = 32
loss_function = "gdl"

## define pl module
class RectalCaSegNet(pl.LightningModule):
    def __init__(self,bs,Height,Depth,epoch_num,l_rate=1e-3,network='resunet',dropout=None,
    channels=model_channels,loss = "gdl", **kwargs):
        super().__init__()
        if network=="deeplab":
            self._model = DeepLabV3_3D(num_classes=2, input_channels=1, resnet='resnet34_os8')
        elif network=="resunet":
            self._model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(model_channels, model_channels*2, model_channels*4, model_channels*8, model_channels*16),
            strides=(2, 2, 2, 2),
            num_res_units=8,
            norm=Norm.BATCH,
            dropout=dropout)
        elif network=="dynunet":
            self._model = DynUNet(spatial_dims=3,in_channels=1,out_channels=2,kernel_size=3,
            strides=[2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            norm_name="batch",
            deep_supervision=True, deep_supr_num=1,res_block=True
            )
        elif network=="kiunet":
            self._model = kiunet3dwcrfb_o(c=1, n=1, num_classes=2)
        elif network=="unet3plus":
            self._model = UNet_3Plus_3D(in_channels=1, n_classes=2, feature_scale=4, 
            is_deconv=True, is_batchnorm=True, channels=model_channels, drop = dropout)

        if loss=="gdl":
            self.loss_function = GeneralizedDiceLoss(
                to_onehot_y=True, softmax=True, sigmoid=False, include_background=False
                )
        elif loss=="focal":
            self.loss_function = FocalLoss(gamma=2.0)
        elif loss == "tversky":
            self.loss_function = TverskyLoss()
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=2)
        self.best_val_metric = 0
        self.best_val_epoch = 0
        self.Width = Height
        self.save_hyperparameters('bs','Height','Depth','epoch_num','l_rate','network','dropout','channels')

    def foward(self, x):
        return self._model(x)
  
  
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr = l_rate, weight_decay=0.001)
        return optimizer

    def training_step(self,batch,batch_idx):
            images, labels = batch["image"], batch["label"]
            output = self.foward(images)
            loss = self.loss_function(output, labels)
            self.log("train_loss",loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return {"loss":loss}

    def validation_step(self,batch,batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (Height, Width, Depth)
        sw_batch_size = 1
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.foward)
        loss = self.loss_function(outputs, labels)
        outputs = self.post_pred(outputs)
        labels = self.post_label(labels)
        metric = compute_meandice(y_pred=outputs, y=labels, include_background=False)
        self.log("val_loss", loss, on_step=False, on_epoch=False, sync_dist=True, prog_bar=False)
        return {"val_loss": loss, "val_metric": metric}

    def validation_epoch_end(self, outputs):
        val_metric, val_loss, num_items = 0, 0, 0
        for output in outputs:
            val_metric += output["val_metric"].sum()
            val_loss += output["val_loss"].sum()
            num_items += len(output["val_metric"])
        mean_val_metric = val_metric / num_items
        mean_val_loss = val_loss / num_items
        self.log("mean_val_loss", mean_val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("mean_val_metric", mean_val_metric, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        if mean_val_metric>self.best_val_metric:
             self.best_val_metric = mean_val_metric
             self.best_val_epoch = self.current_epoch

    def test_step(self,batch,batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (Height, Width, Depth)
        sw_batch_size = 1
        test_image = images
        test_output = sliding_window_inference(test_image, roi_size, sw_batch_size, self.foward)
        loss = self.loss_function(test_output, labels)
        outputs = self.post_pred(test_output)
        label = self.post_label(labels)
        metric = compute_meandice(outputs, label, include_background=False)
        # plot the slice [:, :, rand]
        j = randint(0, len(test_image[0,0,0,0,:])-1)
        plt.figure("check", (20, 4))

        plt.subplot(1, 5, 1)
        plt.title(f"original image {batch_idx}")
        plt.imshow(test_image.detach().cpu()[0, 0, :, :, j], cmap="gray")

        plt.subplot(1, 5, 2)
        plt.title(f"Ground truth mask")
        plt.imshow(labels.detach().cpu()[0, 0, :, :, j])

        plt.subplot(1, 5, 3)
        plt.title(f"AI predicted mask")
        argmax = AsDiscrete(argmax=True)(test_output)
        plt.imshow(argmax.detach().cpu()[0, 0, :, :, j])

        plt.subplot(1, 5, 4)
        plt.title(f"overaying GT")
        map_image1 = test_image.clone().detach()
        map_image1[labels==1] = map_image1.max()
        plt.imshow(map_image1.detach().cpu()[0, 0, :, :, j], cmap="gray")

        plt.subplot(1, 5, 5)
        plt.title(f"overaying predicted")
        map_image2 = test_image.clone().detach()
        map_image2[argmax==1] = map_image2.max()
        plt.imshow(map_image2.detach().cpu()[0, 0, :, :, j], cmap="gray")
        plt.show()
        self.log("test_loss", loss, on_step=False, on_epoch=False, sync_dist=True)
        return {"test_loss": loss, "test_metric": metric}
    def test_epoch_end(self, outputs):
        test_metric, test_loss, num_items = 0, 0, 0
        for output in outputs:
            test_metric += output["test_metric"].sum()
            test_loss += output["test_loss"].sum()
            num_items += len(output["test_metric"])
        mean_test_metric = test_metric / num_items
        mean_test_loss = test_loss / num_items
        self.log("test_metric", mean_test_metric, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", mean_test_loss, on_step=False, on_epoch=True, sync_dist=True)


class RectalCaDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size= batch_size        

    def setup(self, stage=None):
        data_dir = os.path.join(self.data_path, "strata/0/")
        data_dir2 = os.path.join(self.data_path, "strata/1/")
        train_images0 = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii.gz")))
        train_images1 = sorted(glob.glob(os.path.join(data_dir2, "image", "*.nii.gz")))
        train_labels0 = sorted(glob.glob(os.path.join(data_dir, "mask", "*.nii.gz")))
        train_labels1 = sorted(glob.glob(os.path.join(data_dir2, "mask", "*.nii.gz")))
        train_images = train_images0 + train_images1
        train_labels = train_labels0 + train_labels1
        data_dicts0 = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images0, train_labels0)
        ]
        data_dicts1 = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images1, train_labels1)
        ]

        # HistogramStandardization parameter calculation
        histogram_landmarks_path = 'landmarks.npy'
        if os.path.isfile(histogram_landmarks_path):
            print("use landmarks.npy")
        else:
            landmarks = tio.HistogramStandardization.train(train_images,                     
             output_path=histogram_landmarks_path)
            np.set_printoptions(suppress=True, precision=3)

        train_data0, val_data0 = partition_dataset(
            data_dicts0, ratios = [0.8, 0.2], shuffle = True)  
        train_data1, val_data1 = partition_dataset(
            data_dicts1, ratios = [0.8, 0.2], shuffle = True) 
        self.train_data = train_data0 + train_data1
        self.val_data = val_data0 + val_data1
        print('\n'+'Training set:', len(self.train_data), 'subjects')
        print('Validation set:', len(self.val_data), 'subjects')
        #print('Test set:', len(self.test_data), 'subjects')
 
    def train_dataloader(self, *args, **kwargs):
        train_transforms_monai = [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
            ToTensord(keys=["image", "label"]),
        ]
        train_transforms_io = [
            tio.CropOrPad((Height, Width, Depth),mask_name='label', include=["image", "label"]),
            tio.HistogramStandardization({'image': 'landmarks.npy'}, include=["image"]),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
            tio.RandomNoise(p=0.1, include=["image"]),
            tio.RandomFlip(axes=(0,), include=["image", "label"]),
        ]
        train_transforms = Compose(train_transforms_monai + train_transforms_io)

        train_ds = CacheDataset(data=self.train_data, transform=train_transforms, cache_rate=1.0, num_workers=8)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader

    def val_dataloader(self, *args, **kwargs):
        validation_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
        ToTensord(keys=["image", "label"]),
        ]

        validation_transforms_io = [
            tio.CropOrPad((Height, Width, Depth), include=["image", "label"], mask_name='label'),
            tio.HistogramStandardization({'image': 'landmarks.npy'}, include=["image"]),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
        ]
        val_transforms = Compose(validation_transforms_monai + validation_transforms_io)
        val_ds = CacheDataset(data=self.val_data, transform=val_transforms, cache_rate=1.0, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)
        return val_loader

    def test_dataloader(self, *args, **kwargs):
        test_transforms_monai = [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
            ToTensord(keys=["image", "label"]),
        ]
        test_transforms_io = [
        tio.CropOrPad((Height, Width, Depth), include=["image", "label"], mask_name='label'),
        tio.HistogramStandardization({'image': 'landmarks.npy'}, include=["image"]),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
        ]
        test_transforms = Compose(test_transforms_monai + test_transforms_io)
        #test_ds = CacheDataset(data=self.test_data,transform=test_transforms, cache_rate=1.0, num_workers=4)
        test_ds = CacheDataset(data=self.val_data,transform=test_transforms, cache_rate=1.0, num_workers=8)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=8)
        return test_loader


# generate pl data module
dm = RectalCaDataModule(data_path=root_dir, batch_size=bs)

# generate pl training model
net = RectalCaSegNet(bs,Height,Depth,epoch_num,l_rate,network,dropout,model_channels,loss_function)

#setup logger and checkpoints
chk_path = "./checkpoints/"
log_dir = os.path.join(chk_path,"logs")
tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir,log_graph=True)
version = tb_logger.version
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'mean_val_metric',
    mode = 'max',
    dirpath = chk_path,
    filename =f"{network}"+"-{epoch}-{mean_val_loss:.2f}-{mean_val_metric:.2f}-Ver."+f"{version}",
)

#initialize pl trainer
trainer = pl.Trainer(
    gpus=gpu_num,
    max_epochs=epoch_num,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    num_sanity_val_steps=1,
    auto_lr_find=False,
    accelerator='ddp',
    num_nodes=1,
    plugins='ddp_sharded',
    precision=precision,
    profiler="simple"
)

#train start
trainer.fit(net, datamodule=dm)
print(f"train completed, best_metric: {net.best_val_metric:.4f} at epoch: {net.best_val_epoch}")

#test start
trainer.test(model=net, ckpt_path= 'best', datamodule=dm)
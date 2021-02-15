import glob, os
from random import *
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import dice_score
import time, math

import torchio as tio

from monai.config import print_config
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from Ki_UNet_remake import kiunet3dwcrfb_o
from monai.transforms import (
    AsDiscrete, Compose, LoadNiftid, ToTensord, AddChanneld, LabelToContour,
)
from monai.utils import set_determinism

os.environ["MONAI_DATA_DIRECTORY"] = "./data"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = directory
print(root_dir)

### hyperparameter setting
set_determinism(seed=0)

bs = 64
Height = 256
Width = 256
Depth = 32
epoch_num = 450
l_rate = 1e-3
ku = False

## define pl module
class Net(pl.LightningModule):
    def __init__(self,bs,Height,Width,Depth,epoch_num,l_rate, ku):
        super().__init__()
        if ku:
            self._model = kiunet3dwcrfb_o(c=1, n=1, num_classes=2)
        else:
            self._model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=8,
            norm=Norm.BATCH,
            dropout=0.4)

        self.loss_function = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=2)
        self.best_val_metric = 0
        self.best_val_epoch = 0
        self.save_hyperparameters()

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
            tensorboard_logs = {"train_loss": loss}
            return {"loss":loss, "log": tensorboard_logs}

    def validation_step(self,batch,batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (Height, Width, Depth)
        sw_batch_size = 1
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.foward)
        loss = self.loss_function(outputs, labels)
        outputs = self.post_pred(outputs)
        labels = self.post_label(labels)
        #metric = dice_score(outputs,labels)
        metric = compute_meandice(y_pred=outputs, y=labels, include_background=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        #self.log("val_metric", metric, on_step=True, on_epoch=True, sync_dist=True)
        return {"val_loss": loss, "val_metric": metric}

    def validation_epoch_end(self, outputs):
        val_metric, val_loss, num_items = 0, 0, 0
        for output in outputs:
            val_metric += output["val_metric"].sum().item()
            val_loss += output["val_loss"].sum().item()
            num_items += len(output["val_metric"])
        mean_val_metric = val_metric / num_items
        mean_val_loss = val_loss / num_items
        tensorboard_logs = {"val_metric":mean_val_metric, "val_loss":mean_val_loss}
        self.log("val_metric", mean_val_metric, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        if mean_val_metric>self.best_val_metric:
             self.best_val_metric = mean_val_metric
             self.best_val_epoch = self.current_epoch
        return {"log": tensorboard_logs}

    def test_step(self,batch,batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (Height, Width, Depth)
        sw_batch_size = 1
        test_image = images
        test_output = sliding_window_inference(test_image, roi_size, sw_batch_size, self.foward)
        loss = self.loss_function(test_output, labels)
        outputs = self.post_pred(test_output)
        label = self.post_label(labels)
        #metric = dice_score(outputs, label)
        metric = compute_meandice(outputs, label)
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
            test_metric += output["test_metric"].sum().item()
            test_loss += output["test_loss"].sum().item()
            num_items += len(output["test_metric"])
        mean_test_metric = test_metric / num_items
        mean_test_loss = test_loss / num_items
        #tensorboard_logs = {"test_metric":mean_test_metric, "test_loss":mean_test_loss}
        self.log("test_metric", mean_test_metric, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", mean_test_loss, on_step=False, on_epoch=True, sync_dist=True)
        #return {"log": tensorboard_logs}


data_dir = os.path.join(root_dir, "nifti_data")
train_images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

 ### image augmentation transform with monai and torchio API

#HistogramStandardization parameter calculation
histogram_landmarks_path = 'landmarks.npy'
landmarks = tio.HistogramStandardization.train(
     train_images,
     output_path=histogram_landmarks_path,
)
np.set_printoptions(suppress=True, precision=3)

train_data, val_data, test_data = partition_dataset(data_dicts, ratios = [0.8, 0.1, 0.1], shuffle = True)
print('\n'+'Training set:', len(train_data), 'subjects')
print('Validation set:', len(val_data), 'subjects')
print('Test set:', len(test_data), 'subjects')

# transform setting
train_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
]

train_transforms_io = [
        tio.CropOrPad((Height, Width, Depth),mask_name='label', include=["image", "label"]),
        tio.HistogramStandardization({'image': landmarks}, include=["image"]),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
        tio.RandomNoise(p=0.1, include=["image"]),
        tio.RandomFlip(axes=(0,), include=["image", "label"]),
]
train_transforms = Compose(train_transforms_monai + train_transforms_io)
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=1)


validation_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
]

validation_transforms_io = [
    tio.CropOrPad((Height, Width, Depth), include=["image", "label"], mask_name='label'),
    tio.HistogramStandardization({'image': landmarks}, include=["image"]),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
]
val_transforms = Compose(validation_transforms_monai + validation_transforms_io)
val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=1)

test_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
]
test_transforms_io = [
    tio.CropOrPad((Height, Width, Depth), include=["image", "label"], mask_name='label'),
    tio.HistogramStandardization({'image': landmarks}, include=["image"]),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
]
test_transforms = Compose(test_transforms_monai + test_transforms_io)
test_ds = CacheDataset(data=test_data,transform=test_transforms, cache_rate=1.0, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)



# generate pl module
net = Net(bs,Height,Width,Depth,epoch_num,l_rate,ku)

#setup logger and checkpoints
chk_path = "./checkpoints"
log_dir = os.path.join(chk_path,"logs")
tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    filepath=os.path.join(log_dir, "{epoch}-{val_loss:.2f}-{val_metric:.2f}")
)

#inti pl trainer
trainer = pl.Trainer(
    gpus=4,
    max_epochs=epoch_num,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    num_sanity_val_steps=1,
    auto_lr_find=False,
    accelerator='ddp',
    num_nodes=1,
    plugins='ddp_sharded',
    precision=16
)

#train
trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
print(f"train completed, best_metric: {net.best_val_metric:.4f} at epoch: {net.best_val_epoch}")

trainer.test(net, test_dataloaders=test_loader)
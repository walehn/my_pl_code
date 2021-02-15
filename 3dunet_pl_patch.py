import copy, os
import enum
from random import *
import warnings
import tempfile
import subprocess
import multiprocessing
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import random; random.seed(seed)

import torchio as tio
from torchio import AFFINE, DATA

import numpy as np
import nibabel as nib
from unet import UNet
from scipy import stats
import SimpleITK as sitk
import matplotlib.pyplot as plt

from IPython import display
from tqdm.notebook import tqdm
import pytorch_lightning as pl

print('TorchIO version:', tio.__version__)

num_epochs = 100
l_rate = 1e-3


# Dataset
dataset_dir_name = './data/nifti_data'
dataset_dir = Path(dataset_dir_name)

images_dir = dataset_dir / 'image'
labels_dir = dataset_dir / 'mask'
image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))
assert len(image_paths) == len(label_paths)

subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        mask=tio.LabelMap(label_path),
    )
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

histogram_landmarks_path = 'landmarks.npy'
landmarks = tio.HistogramStandardization.train(
    image_paths,
    output_path=histogram_landmarks_path,
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained landmarks:', landmarks)

training_transform = tio.Compose([
    #tio.ToCanonical(),
    #tio.Resample(3),
    tio.CropOrPad((128, 128, 16),mask_name='mask'),
    tio.RandomMotion(p=0.2),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.RandomBiasField(p=0.1),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.1),
    tio.RandomFlip(axes=(0,)),
    tio.RandomAffine(),
])

validation_transform = tio.Compose([
    #tio.ToCanonical(),
    #tio.Resample(3),
    tio.CropOrPad((128, 128, 16), mask_name='mask'),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

training_split_ratio = 0.8  # use 80% of samples for training, 20% for testing
test_split_ratio = 0.5

num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)
training_subjects = subjects[:num_training_subjects]

validation_subjects = subjects[num_training_subjects:]
num_test_subjects = int(len(validation_subjects)*0.5)
validation_subjects = validation_subjects[num_test_subjects:]
test_subjects = validation_subjects[:num_test_subjects]

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

test_set = tio.SubjectsDataset(
    test_subjects, transform=validation_transform
)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')
print('Test set:', len(test_set), 'subjects')

training_batch_size = 32
validation_batch_size = 2 * training_batch_size

patch_size = (64,64,8)
samples_per_volume = 16
max_queue_length = 1024

patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=tio.data.UniformSampler(patch_size),
    num_workers=0,
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=tio.data.UniformSampler(patch_size),
    num_workers=0,
    shuffle_subjects=False,
    shuffle_patches=False,
)

training_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

one_batch = next(iter(training_loader))
k = int(patch_size[2] // 4)
batch_mri = one_batch['mri'][DATA][..., k]
batch_label = one_batch['mask'][DATA][..., k]
slices = torch.cat((batch_mri, batch_label))
image_path = 'batch_patches.png'
save_image(slices, image_path, nrow=training_batch_size, normalize=True, scale_each=True)
display.Image(image_path)


class Net(pl.LightningModule):
    def __init__(self,bs,Height,Width,Depth,epoch_num,l_rate):
        super().__init__()
        self._model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    )
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


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, device):
    inputs = batch['mri'][DATA].to(device)
    foreground = batch['mask'][DATA].to(device)
    background = 1 - foreground
    targets = torch.cat((background, foreground), dim=CHANNELS_DIMENSION)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits

def get_model_and_optimizer(device):
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    return epoch_losses.mean()

def train(num_epochs, training_loader, validation_loader, model, optimizer):
    training_loss_values = []
    validation_loss_values = []
    best_val_loss = 1
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
    for epoch_idx in range(1, num_epochs + 1):
        training_loss = 0.0
        val_loss = 0.0
        print('Starting epoch', epoch_idx)

        lv_t = run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
        training_loss += lv_t
        training_loss_values.append(training_loss)

        lv_v = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
        val_loss += lv_v
        validation_loss_values.append(val_loss)
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dataset_dir, "best_metric_model.pth"))
                print("saved new best metric model")

    plt.figure(figsize=(12,5))    
    plt.plot(training_loss_values)
    plt.plot(validation_loss_values)
    plt.title("3D Unet Model Loss")
    plt.ylabel("loss")
    plt.xlabel("Epochs")
    plt.legend(['train', 'val'])


model, optimizer = get_model_and_optimizer(device)
train(num_epochs, training_loader, validation_loader, model, optimizer)
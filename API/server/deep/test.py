import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as plt_path
import cv2 
from PIL import Image

from pathlib import Path
from pycocotools.coco import COCO

DATASET_PATH = Path('/home/tuananh/Documents/build_on/dataset')
img_root_path = DATASET_PATH / 'train/images'

import albumentations as albu
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Lambda
    )
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

from torchvision.io import read_image
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
def get_mask(coco, img_id, cat_ids):
    img_inf = coco.loadImgs(img_id)[0]
    img_shape = (img_inf['width'], img_inf['height'], len(cat_ids))
    ann_ids = coco_anno.getAnnIds(imgIds=img_id,
                                  catIds=cat_ids,
                                  iscrowd=None)
    anns = coco.loadAnns(ann_ids)
#     masks = np.zeros((img_inf['height'],img_inf['width'], len(cat_ids)))
#     for idx, cat_id in enumerate(cat_ids):
#         mask = np.zeros((img_inf['height'],img_inf['width']))
#         for ann in anns:
#             if cat_id == ann['category_id']:
# #                 mask = coco_anno.annToMask(ann)
#                 mask = np.maximum(mask, coco_anno.annToMask(ann))
#         masks[:, :, idx] = mask
    masks = np.zeros(img_shape)
    for idx, cat_id in enumerate(cat_ids):
        mask = np.zeros(img_shape[:2])
        for ann in anns:
            if cat_id == ann['category_id']:
                mask = np.maximum(mask, coco_anno.annToMask(ann))
        masks[:, :, idx] = mask
    return masks


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    

seed_torch()

def get_training_augmentation(width=320, height=320):
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        albu.RandomCrop(height=height, width=width, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)
    
    
def get_validation_augmentation(width=320, height=320):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
#         albu.PadIfNeeded(min_height=height, min_width=width)
        albu.Resize(width, height)
    ]
    return albu.Compose(test_transform)


def get_transform(resize, phase='train'):
    _transforms = []
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    if phase == 'train':
        _transforms.append(Resize(resize, resize))
        _transforms.append(HorizontalFlip(p=0.5))
        _transforms.append(VerticalFlip(p=0.5))
#         for t in [
#             RandomResizedCrop(resize, resize),
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ShiftScaleRotate(p=0.5)]:
#             _transforms.append(t)
        
    else:
        _transforms.append(Resize(resize, resize))
    # 
    _transforms.append(Normalize(
                mean=_mean,
                std=_std,
            ))
    _transforms.append(ToTensorV2())
    return Compose(_transforms)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor)
    ]
    return Compose(_transform)


class SegDataset(Dataset):
    def __init__(self, root_dir, img_ids, cat_ids, coco_api, transforms=None, preprocessing=None):
        self.root_dir = root_dir
        self.img_ids = img_ids
        self.cat_ids = cat_ids
        self.coco_api = coco_api
        self.transforms = transforms
        self.preprocessing = preprocessing
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_inf = self.coco_api.loadImgs(img_id)[0]
        file_name = img_inf['file_name']
        file_path = self.root_dir / file_name
        img = Image.open(file_path).convert('RGB')
        mask = get_mask(self.coco_api, img_id, self.cat_ids)
        
        if self.transforms:
            augmented = self.transforms(image=np.array(img), mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if self.preprocessing:
            augmented = self.preprocessing(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask
    
    def __len__(self):
        return len(self.img_ids)

anno_path = DATASET_PATH / 'annotation-small.json'
with open(DATASET_PATH / 'annotation-small.json', 'r') as f:
    annot_data = json.load(f)
    
coco_anno = COCO(str(anno_path))
cat_ids = coco_anno.getCatIds()
cats = coco_anno.loadCats(cat_ids)
img_ids = coco_anno.getImgIds(catIds=cat_ids)

train_img_ids, valid_img_ids = train_test_split(img_ids, test_size=0.1, random_state=42)
valid_img_ids, test_img_ids = train_test_split(valid_img_ids, test_size=0.5, random_state=42)
print(len(train_img_ids), len(valid_img_ids), len(test_img_ids))

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

_preprocessing = get_preprocessing(preprocessing_fn)

test_dataset = SegDataset(root_dir=img_root_path,
                           img_ids=test_img_ids,
                           cat_ids=cat_ids,
                           coco_api=coco_anno,
                           transforms=get_validation_augmentation(288, 288),
                           preprocessing=_preprocessing)

test_dataloader = DataLoader(test_dataset)



best_model = torch.load('/home/tuananh/Documents/build_on/epoch/best_model_Unet_resnet50.pth')

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True
)

logs = test_epoch.run(test_dataloader)

for i in range(3):
    n = np.random.choice(len(test_dataset))
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

    pr_mask = best_model.predict(x_tensor)

    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    cnts = cv2.findContours(pr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for c in cnts:
        M = cv2.moments(c)
        centroidX = int(M["m10"] / M["m00"])
        centroidY = int(M["m01"] / M["m00"])
        cv2.circle(image, (centroidX, centroidY), 5, (0,255,0), -1)

        s = cv2.contourArea(c)
    print(s)

    visualize(
        image=image.transpose(1, 2, 0), 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )

    

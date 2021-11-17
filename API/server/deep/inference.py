import json
import os
import random
import numpy as np
import cv2 
from PIL import Image
import albumentations as albu
from albumentations import (
    Compose, Lambda
    )
import segmentation_models_pytorch as smp
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)






def get_validation_augmentation(width=320, height=320):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
#         albu.PadIfNeeded(min_height=height, min_width=width)
        albu.Resize(width, height)
    ]
    return albu.Compose(test_transform)





def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor)
    ]
    return Compose(_transform)



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


transformer = get_validation_augmentation(288, 288)





best_model = torch.load('./unet.pth', map_location=torch.device('cpu'))




img = Image.open('./land2.jpg').convert('RGB')


test = np.array(img)


augmented = transformer(image=test)
img = augmented['image']

print(img.shape)

augmented = _preprocessing(image=img)
img = augmented['image']


x_tensor = torch.from_numpy(img).to(device).unsqueeze(0)

pr_mask = best_model.predict(x_tensor)
pred = pr_mask.detach().cpu().numpy()
pred=pred[0,:,:,:]*255
pred=pred.transpose(1,2,0)
pred=np.where(pred < 125, 0, 255)
pred=pred[:,:,-1]
pred=pred.astype(np.uint8)


print(pred.shape)

cv2.imshow('pred', pred)
cv2.waitKey(1000)

contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
for c in contours:
    s = cv2.contourArea(c)
    print(s)


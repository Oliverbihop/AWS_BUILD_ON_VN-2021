import json
import os
import random
from symbol import argument
import numpy as np
import cv2 
from PIL import Image
import albumentations as albu
from albumentations import (
    Compose, Lambda
    )
import segmentation_models_pytorch as smp
import torch
import base64
from io import BytesIO


class Unet():
    def __init__(self):
        self.model = torch.load('./server/deep/unet.pth', map_location=torch.device('cpu'))
        
        ENCODER = 'resnet50'
        ENCODER_WEIGHTS = 'imagenet'
        
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        
        self.transform = albu.Compose([
            albu.Resize(288, 288)
        ])


        self.preprocessing = self.get_pre_processing(preprocessing_fn)
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def base64_to_image(self, base64_data):
        encoded_data = base64_data
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        return img

    def get_pre_processing(self, preprocessing_fn):
        _transform = [
            Lambda(image=preprocessing_fn),
            Lambda(image=self.to_tensor, mask=self.to_tensor)
        ]
        return Compose(_transform)


    def predict(self, hashImage):
        image = self.base64_to_image(hashImage)
        argument = self.transform(image=image)
        image = argument['image']
        print(image.shape)
        argument = self.preprocessing(image=image)
        image = argument['image']
        print(image.shape)

        x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)

        pr_mask = self.model.predict(x_tensor)
        pred = pr_mask.detach().cpu().numpy()
        pred=pred[0,:,:,:]*255
        pred=pred.transpose(1,2,0)
        pred=np.where(pred < 125, 0, 255)
        pred=pred[:,:,-1]
        pred=pred.astype(np.uint8)


        area = 0
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("Number of Contours found = " + str(len(contours)))
        for c in contours:
            s = cv2.contourArea(c)
            area += s

        pil_img = Image.fromarray(pred)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        return {
            'image' : new_image_string,
            'area' : area,
            'building' : len(contours)
        }




    
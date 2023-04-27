import configs
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils.datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import logging
import utils.type_converter as tcvt
from utils.type_converter import COCO_converter
import json
import torch.nn.functional as F
import logging
from datetime import datetime  
import time 
from functools import partial
from utils.logger import create_logger

def min_max_scaler(x, min_v, max_v):
    return (x - min_v) / (max_v - min_v)

logger = create_logger()

scaler =  partial(min_max_scaler , min_v = 23.580, max_v = 38.902)
train_transformer = A.Compose([
    A.Resize(height=224, width=224),
    A.HorizontalFlip(),
    A.Rotate(),
    # A.Normalize(mean=0.5, std=0.5,
    #             max_pixel_value=255.0),
    ToTensorV2(),
],)


val_transformer = A.Compose([
    A.Resize(height=224, width=224),
    # A.Normalize(mean=0.5, std=0.5,
    #             max_pixel_value=255.0),
    ToTensorV2(),
],)
val_dataset = utils.datasets.RegCustomDataset(coco_dir='./dataset/val.json' , 
                                              celcius_prefix='./dataset/celsius/', 
                                              metas_df_path = './dataset/img_metas.csv', 
                                              transformer = val_transformer  , 
                                              label_type = 'eyes')

val_dataloader = DataLoader(dataset=val_dataset, 
                            batch_size=16 , 
                            pin_memory= True)

val_coco_cvt = COCO_converter('./dataset/val.json')


model = configs.reg_resnet.Reg_ResNet50()

# pretrained
pretrained_weights = torch.load('./ckpts/non_normalize_test/epoch_200.pth')
model.load_state_dict(pretrained_weights , strict = False)

epochs = 200
lr = 6 * 1e-5
weight_decay = 0.01
warm_up_step = 100
interval = 50
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=lr)


save_name = 'non_normalize_test'
device = torch.device('cuda')
os.makedirs(f'./result/{save_name}' ,exist_ok= True)
model = model.to(device)


losses = 0
result_dict = {}
with torch.no_grad():
    model.eval()
    for i , ( img_metas , images , temps) in enumerate(val_dataloader):
        images = images.to(device)
        images = scaler(images)
        temps = temps.to(device)
        preds = model(images)
        preds =preds.squeeze()
        loss = criterion(preds, temps)
        losses += loss.item()
        result_dict.update(zip(img_metas['file_name'] , preds.cpu().numpy()))
with open(f'./result/{save_name}/pred_result.json', "w") as json_file:
    json.dump([result_dict], json_file, indent=4 , cls = tcvt.NpEncoder)

logger.info(f'val_losses , {losses / len(val_dataloader)}')
logger.info(temps[:5])
logger.info(preds[:5])

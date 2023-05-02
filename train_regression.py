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
from utils.logger import create_logger
from functools import partial
from timm.scheduler.cosine_lr import CosineLRScheduler

def min_max_scaler(x, min_v, max_v):
    return (x - min_v) / (max_v - min_v)

save_name = '4_category_epoch1000_AdamW_lr_1e-4_consine'
device = torch.device('cuda')
os.makedirs(f'./ckpts/{save_name}' ,exist_ok= True)
os.makedirs(f'./result/{save_name}' ,exist_ok= True)
resume_start = False
logger = create_logger(save_name)
label_type = None

scaler =  partial(min_max_scaler , min_v = 23.580, max_v = 38.902)
train_transformer = A.Compose([
    A.Resize(height=224, width=224),
    A.HorizontalFlip(),
    # A.Rotate(),
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
train_dataset = utils.datasets.RegCustomDataset(coco_dir='./dataset/train.json', 
                                                celcius_prefix='./dataset/celsius/', 
                                                metas_df_path = './dataset/img_metas.csv' , 
                                                transformer = train_transformer,
                                                label_type= label_type
                                                )

val_dataset = utils.datasets.RegCustomDataset(coco_dir='./dataset/val.json' , 
                                              celcius_prefix='./dataset/celsius/', 
                                              metas_df_path = './dataset/img_metas.csv', 
                                              transformer = val_transformer ,
                                              label_type= label_type
                                              )
batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size , pin_memory= True )
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size , pin_memory= True)
val_coco_cvt = COCO_converter('./dataset/val.json')


model = configs.reg_resnet.Reg_ResNet152(num_inputs= 4)

# pretrained

if resume_start:
    pretrained_weights = torch.load('./ckpts/eyes_test/epoch_10.pth')
else:
    pretrained_weights = torch.load('./ckpts/pretrained/resnet152-394f9c45.pth')
    pretrained_weights.pop('conv1.weight')
    pretrained_weights.pop('fc.weight')
    pretrained_weights.pop('fc.bias')

model.load_state_dict(pretrained_weights , strict = False)

epochs = 500
lr = 1e-4
interval = 50
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=lr)
lr_scheduler = CosineLRScheduler(
    optimizer,
    t_initial=(2000 - 500),
    lr_min=5e-6,
    warmup_lr_init=5e-7,
    warmup_t=1000,
    cycle_limit=20,
    t_in_epochs=False,
    warmup_prefix=True,
)




model = model.to(device)
step = 1
for epoch in range(1, epochs + 1):
    losses = 0
    model.train()
    logger.info(f'epoch : {epoch}')
    
    for i , (img_metas, images , temps) in enumerate(train_dataloader,1):
        images = images.to(device)
        images = scaler(images)
        temps = temps.to(device)
        temps = temps.type(torch.float32)
        preds = model(images)
        preds =preds.squeeze()
        loss = criterion(preds, temps)
        losses += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update(num_updates=step)
        step += 1
        if i % interval == 0:
            logger.info(f'[{i} / {len(train_dataloader)}]')
            logger.info(f'train loss {i} : {losses / interval}')
            losses = 0

    if epoch:
        result_dict = {}
        losses = 0
        with torch.no_grad():
            model.eval()
            for i , (img_metas, images, temps) in enumerate(val_dataloader):
                images = images.to(device)
                images = scaler(images)
                temps = temps.to(device)
                preds = model(images)
                preds =preds.squeeze()
                loss = criterion(preds, temps)
                losses += loss.item()
                print(loss)
                result_dict.update(zip(img_metas['file_name'] , preds.cpu().numpy()))
        with open(f'./result/{save_name}/pred_result_{epoch}.json', "w") as json_file:
            json.dump([result_dict], json_file, indent=4 , cls = tcvt.NpEncoder)

        logger.info(f'preds : {preds[:10]}')
        logger.info(f'temps : {temps[:10]}')
        logger.info(f'val_losses : {losses / len(val_dataloader)}')
        
    if epoch % 10 == 0: 
        logger.info(f'epoch : {epoch} saved ckpts')
        torch.save(model.state_dict(), f'./ckpts/{save_name}/epoch_{epoch}.pth')
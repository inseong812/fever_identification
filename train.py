import model
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

backbone = model.resnet.ResNet50()
dec_head = model.unet.UNetDEC()

# pretrained
pretrained_weights = torch.load('./ckpts/pretrained/resnet50-19c8e357.pth')
pretrained_weights.pop('fc.weight')
pretrained_weights.pop('fc.bias')
backbone.load_state_dict(pretrained_weights)

class Seg_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.dec_head = dec_head

    def forward(self, x):
        out = self.backbone(x)
        out = self.dec_head(out)
        return out

model = Seg_model()

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
de_std = tuple(std * 255 for std in std)
de_mean = tuple(mean * 255 for mean in mean)

transformer = A.Compose([
    # A.Resize(height=512, width=512),
    A.Normalize(mean=0.5, std=0.5,
                max_pixel_value=255.0),
    ToTensorV2(),
],)

train_dataset = utils.datasets.CustomDataset( './dataset/train.json', img_prefix = './dataset/img/' ,transformer=transformer)
val_dataset = utils.datasets.CustomDataset( './dataset/test.json', img_prefix = './dataset/img/' ,transformer=transformer)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16 , pin_memory= True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16 , pin_memory= True)


epochs = 50
lr = 1e-3

# model.load_state_dict(torch.load('./ckpts/data_retina_num_5000_epoch_100.pth'))
model = model.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),
                        lr=0.0001 / 8,
                        betas=(0.9, 0.999),
                        weight_decay=0.05,
                        )

save_name = 'test_AdamW'
os.makedirs(f'./ckpts/{save_name}' ,exist_ok= True)

for epoch in range(1, epochs + 1):
    model.train()
    print(f'epoch : {epoch}')
    losses = 0
    for i , (images , masks) in enumerate(train_dataloader):
        images = images.cuda()
        masks = masks.cuda()
        preds = model(images)
        preds = preds.squeeze()
        loss = criterion(preds, masks)
        losses += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 50 == 0 and i > 0 :
            print(f'[{i} / {len(train_dataloader)}]')
            print(f'train loss {i} :' , losses / 100)
    if epoch:
        losses = 0
        with torch.no_grad():
            model.eval()
            for i , (images , masks) in enumerate(val_dataloader):
                images = images.cuda()
                masks = masks.cuda()
                preds = model(images)
                preds = preds.squeeze()
                loss = criterion(preds, masks)
                losses += loss.item()
            print('val_losses' , losses / len(val_dataloader))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'./ckpts/{save_name}/epoch_{epoch}.pth')












# if __name__=="__main__":
#     model = Seg_model()
#     inputs = torch.randn(1,3,480,640)
#     outputs = model(inputs)
#     torch.onnx.export(model,  inputs, 'resnet-unet.onnx', 
#                     input_names = ['inputs'] , 
#                     output_names = ['preds'], opset_version = 11 
#                   )
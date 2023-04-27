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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

backbone = configs.resnet.ResNet50()
dec_head = configs.unet.UNetDEC(output_channel=3)

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

train_transformer = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(),
    A.Rotate(),
    A.Normalize(mean=0.5, std=0.5,
                max_pixel_value=255.0),
    ToTensorV2(),
],)

val_transformer = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=0.5, std=0.5,
                max_pixel_value=255.0),
    ToTensorV2(),
],)
def collate_fn(batch):
    image_list = []
    masks_list = []
    img_metas = []
    for img_meta, image, mask in batch:
        image_list.append(image)
        img_metas.append(img_meta)
        masks_list.append(mask)

    return img_metas, torch.stack(image_list, dim=0), torch.stack(masks_list, dim=0)

train_dataset = utils.datasets.CustomDataset( './dataset/train.json', img_prefix = './dataset/img/' ,transformer=train_transformer)
val_dataset = utils.datasets.CustomDataset( './dataset/val.json', img_prefix = './dataset/img/' ,transformer=val_transformer)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16 , pin_memory= True , collate_fn=collate_fn)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16 , pin_memory= True, collate_fn=collate_fn)
val_coco_cvt = COCO_converter('./dataset/val.json')

epochs = 1000
lr = 1e-3
interval = 50
threshold = 0.5
# model.load_state_dict(torch.load('./ckpts/data_retina_num_5000_epoch_100.pth'))


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(model.parameters(),
                        lr=lr
                        )
dice_criterion = configs.DiceLoss()
save_name = 'test_dice'
device = torch.device('cuda')
os.makedirs(f'./ckpts/{save_name}' ,exist_ok= True)
model = model.to(device)
losses = 0
for epoch in range(1, epochs + 1):
    model.train()
    print(f'epoch : {epoch}')
    
    for i , (img_metas, images , masks) in enumerate(train_dataloader,1):
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        preds = preds.squeeze()
        loss = criterion(preds, masks)
        loss += dice_criterion(preds, masks)
        losses += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % interval == 0:
            print(f'[{i} / {len(train_dataloader)}]')
            print(f'train loss {i} :' , losses / interval)
            losses = 0
    if epoch:
        result_list = list()
        losses = 0
        with torch.no_grad():
            model.eval()
            for i , (img_metas, images , masks) in enumerate(val_dataloader):
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                preds = preds.squeeze()
                loss = criterion(preds, masks)
                loss += dice_criterion(preds, masks)
                losses += loss.item()
                preds = F.sigmoid(preds) > threshold
                result = val_coco_cvt.preds_to_json(img_metas , preds)
                result_list.extend(result)
        with open(f'./result/{save_name}/result_{epoch}.json', "w") as json_file:
            json.dump(result_list, json_file, indent=4 , cls = tcvt.NpEncoder)

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
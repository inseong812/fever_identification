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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

backbone = model.resnet.ResNet50()
dec_head = model.unet.UNetDEC()

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

val_dataset = utils.datasets.CustomDataset( './dataset/test.json', img_prefix = './dataset/img/' ,transformer=transformer)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1 , pin_memory= True)


model.load_state_dict(torch.load('./ckpts/test/epoch_10.pth'))
model = model.cuda()
criterion = nn.BCEWithLogitsLoss()

losses = 0
threshold = 0.5
save_name = 'test'
os.makedirs(f'./result/{save_name}' ,exist_ok= True)

with torch.no_grad():
    model.eval()
    for i , (images , masks) in enumerate(val_dataloader):
        images = images.cuda()
        masks = masks.cuda()
        preds = model(images)
        preds = preds.squeeze()
        masks = masks.squeeze()
        loss = criterion(preds, masks)
        pred_masks = F.sigmoid(preds) > threshold
        losses += loss.item()
        img = images.squeeze()

        img = img.permute(1,2,0).cpu()
        masks = masks.cpu()
        pred_masks = pred_masks.cpu()

        img = ((img *  torch.tensor(de_std)) + torch.tensor(de_mean)).int().numpy()
        fig , ax = plt.subplots(1,3, figsize = (16,5))
        ax[0].imshow(img)
        ax[1].imshow(masks.numpy())
        ax[2].imshow(pred_masks.float().numpy())
        fig.savefig(f'./result/{save_name}/{i}.jpg')

        if i % 100 ==0 and i >0: break
    print('val_losses' , losses / len(val_dataloader))














# if __name__=="__main__":
#     model = Seg_model()
#     inputs = torch.randn(1,3,480,640)
#     outputs = model(inputs)
#     torch.onnx.export(model,  inputs, 'resnet-unet.onnx', 
#                     input_names = ['inputs'] , 
#                     output_names = ['preds'], opset_version = 11 
#                   )
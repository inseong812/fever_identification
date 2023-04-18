from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import matplotlib.pyplot as plt
import math
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy


class CustomDataset(data.Dataset):
    def __init__(self, coco_dir, img_prefix,  transformer=None):
        super(CustomDataset, self).__init__()
        self.coco = COCO(coco_dir)
        self.img_prefix = img_prefix
        self.img_idx_list = self.coco.getImgIds()
        self.transformer = transformer

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        mask  = self.get_label(idx)

        if self.transformer:
            transformed_data = self.transformer(image=image, mask=mask)
            image = transformed_data['image']
            mask = transformed_data['mask']
        return  image, mask.float()

    def get_image(self, idx):
        idx = self.img_idx_list[idx]
        img_info = self.coco.loadImgs(ids=idx)[0]
        filename = img_info['file_name']
        image_path = os.path.join(self.img_prefix, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_label(self, idx):
        idx = self.img_idx_list[idx]
        mask = self.coco.annToMask(self.coco.loadAnns(idx)[0]) # 임시적인 mask
        return mask


if __name__=='__main__':
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

    dataset = CustomDataset( './dataset/coco.json', img_prefix = './dataset/img/' ,transformer=transformer)
    dataloader = DataLoader(dataset=dataset, batch_size=16)
    inputs = next(iter(dataloader))
    for i , (images , masks) in enumerate(dataloader):
        pass
    fig , ax =plt.subplots(1,2, figsize = (16,5))
    img , mask = inputs[0][0] , inputs[1][0].numpy()
    img = img.permute(1,2,0)
    img = ((img *  torch.tensor(de_std)) + torch.tensor(de_mean)).int().numpy()
    ax[0].imshow(img)
    ax[1].imshow(mask)
    fig.savefig('./temp/input_debug.jpg')
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
import os.path as osp 
import pandas as pd 
import numpy as np
import re 
import utils.type_converter as tcvt


class CustomDataset(data.Dataset):
    '''
    Args:
        coco_dir [str] : coco type의 json file path
        img_prefix [str] : image folder path
        transformer Type [albumentations] :  
    Example :
        >>> transformer = A.Compose([
                            A.Normalize(mean=0.5, std=0.5,
                            max_pixel_value=255.0),
                            ToTensorV2(),],)
        >>> dataset = CustomDataset( './dataset/coco.json', img_prefix = './dataset/img/' ,transformer=transformer)
        >>> dataset[3]
    '''
    def __init__(self, coco_dir, img_prefix,  transformer=None):
        super(CustomDataset, self).__init__()
        self.coco = COCO(coco_dir)
        self.img_prefix = img_prefix
        self.img_idx_list = self.coco.getImgIds()
        self.transformer = transformer

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        img_metas, image = self.get_image(idx)
        mask  = self.get_label(idx)

        if self.transformer:
            transformed_data = self.transformer(image=image, mask=mask)
            image = transformed_data['image']
            mask = transformed_data['mask']
            mask = mask.permute(2,0,1)
        return  img_metas, image.type(torch.float32), mask.type(torch.float32)

    def get_image(self, idx):
        idx = self.img_idx_list[idx]
        img_info = self.coco.loadImgs(ids=idx)[0]
        filename = img_info['file_name']
        image_path = os.path.join(self.img_prefix, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {'ori_img_size' : image.shape , 'file_name' : filename} , image

    def get_label(self, idx):
        idx = self.img_idx_list[idx]
        catIds = len(self.coco.getCatIds())
        mask_list = []
        for i in range(1,catIds+1):
            mask_annos = self.coco.loadAnns(self.coco.getAnnIds(imgIds = idx, catIds = i))
            if len(mask_annos) > 1:
                mask = self.coco.annToMask(mask_annos[0])
                for mask_anno in mask_annos[1:]:
                    mask_i = self.coco.annToMask(mask_anno)
                    mask[mask_i > 0 ] = 1
            else:
                mask = self.coco.annToMask(mask_annos[0]) # 임시적인 mask
            mask_list.append(mask)
        return np.stack(mask_list , 2)

class RegCustomDataset(data.Dataset):
    '''
    Args:
        coco_dir [str] : coco type의 json file path
        img_prefix [str] : image folder path
        transformer Type [albumentations] :  
    Example :
        >>> transformer = A.Compose([
                            A.Normalize(mean=0.5, std=0.5,
                            max_pixel_value=255.0),
                            ToTensorV2(),],)
        >>> dataset = CustomDataset( './dataset/coco.json', img_prefix = './dataset/img/' ,transformer=transformer)
        >>> dataset[3]
    '''
    def __init__(self, coco_dir, celcius_prefix, metas_df_path ,transformer=None  , label_type = 'eyes'):
        super(RegCustomDataset, self).__init__()
        self.coco_gt = COCO(coco_dir)
        self.celcius_prefix = celcius_prefix
        self.img_idx_list = self.coco_gt.getImgIds()
        self.label_df = pd.read_csv(metas_df_path)
        self.transformer = transformer

        self.img_name_to_id = {os.path.basename(image_info['file_name']): image_info['id'] for image_info in self.coco_gt.loadImgs(self.coco_gt.getImgIds())}
        self.imgId_to_name = {v: k for k, v in self.img_name_to_id.items()}
        self.cat_name_to_id = {cat_info['name']: cat_info['id'] for cat_info in self.coco_gt.loadCats(self.coco_gt.getCatIds())}
        self.catId_to_name = {v: k for k, v in self.cat_name_to_id.items()}
        self.cat_nums = len(self.coco_gt.getCatIds())
        if label_type:
            self.label_type = label_type
            keys = list(self.cat_name_to_id.keys())
            self.aval_idx = np.nonzero(np.where( np.array(keys) == label_type ))[0][0]
    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        img_metas , image = self.get_celcius(idx)
        temp  = self.get_label(idx)

        if self.transformer:
            transformed_data = self.transformer(image=image)
            image = transformed_data['image']
        if  image.dtype != torch.float32: image = image.type(torch.float32)
        if not torch.is_tensor(temp) or temp.dtype != torch.float32: temp = torch.tensor(temp , dtype = torch.float32)

        return  img_metas , image , temp

    def get_celcius( self  , idx ):        
        
        idx = self.img_idx_list[idx]
        img_name = self.imgId_to_name[idx]
        npy_name = img_name.replace('jpg' , 'npy')
        celsius_path= osp.join(self.celcius_prefix, npy_name)
        with open(celsius_path, 'rb') as f:
            celsius = np.load(f, encoding ='ASCII') 
        one_hot_mask = tcvt.get_one_hot_mask(self.coco_gt , idx = idx)
        if self.label_type:
            mask = one_hot_mask[...,self.aval_idx]
        else:
            mask = np.max(one_hot_mask , axis =2)

        celsius = celsius * mask
        x1, y1, x2, y2 = np.nonzero(celsius)[0].min(),np.nonzero(celsius)[1].min() , \
                np.nonzero(celsius)[0].max(), np.nonzero(celsius)[1].max()
        return {'file_name' : img_name } , celsius[x1-1:x2+1,y1-1:y2+1]

    def get_label(self, idx):
        idx = self.img_idx_list[idx]
        img_name = self.imgId_to_name[idx]
        flyr_id = int(re.findall('\d+' , img_name)[0])
        temperature = self.label_df['ear_temp'][self.label_df['No.'] ==flyr_id].values[0]
        return temperature
    
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
    dataset[3]
    len(dataset)
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
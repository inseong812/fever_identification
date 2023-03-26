import os
import numpy as np
import shutil
import cv2
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import random
import argparse
from collections import defaultdict
import re


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


warnings.filterwarnings(action='ignore')

# parser = argparse.ArgumentParser(
#     description='labelme의 creatML format을 cocoformat으로 바꾸는 script'
# )

# parser.add_argument('--label_path')
# parser.add_argument('--sample_num', type=int, default=0)
# parser.add_argument('--save_path')

# args = parser.parse_args()

label_path_list = glob.glob('../dataset/label/*')

# make img dataframe
df_img = pd.DataFrame(
    columns=['id', 'file_name', 'RESOLUTION', 'width', 'height'])

# make box object dataframe
df_obj = pd.DataFrame(
    columns=['id', 'category_id', 'bbox', 'image_id', 'segmentation'])

obj_idx = 1
cats = set()
for idx, label_path in enumerate(label_path_list, start=1):
    # json to img file path
    img_path = label_path.replace('label', 'img').replace('json', 'jpg')

    # load json file
    with open(label_path, 'r') as file:
        json_file = json.load(file)
    df_img = df_img.append(
        {'id': idx,
         'file_name': img_path,
         'RESOLUTION': json_file['imageHeight'] * json_file['imageWidth'],
         'height': json_file['imageHeight'],
         'width': json_file['imageWidth']}, ignore_index=True
    )

    # get object
    obj_list = json_file['shapes']
    for obj in obj_list:
        label = obj['label']
        cats.add(label)
        anno = np.array(obj['points']).astype(int)
        x1, y1, x2, y2 = [*anno.min(axis=0), *anno.max(axis=0)]
        bbox = [x1, y1, x2-x1, y2 - y1]
        segmentation = [anno.reshape(-1).astype(int).tolist()]
        df_obj = df_obj.append(
            {'id': obj_idx,
             'category_id': label,
             'bbox': bbox,
             'segmentation': segmentation,
             'image_id': idx}, ignore_index=True
        )
        obj_idx += 1


random.seed(555)


df_obj['area'] = df_obj['bbox'].apply(lambda x: x[2] * x[3])
cats_id = defaultdict()
for i, cat in enumerate(cats, start=1):
    cats_id[cat] = i

df_obj['category_id'] = df_obj['category_id'].apply(lambda x: cats_id[x])
df_obj['iscrowd'] = 0

# cvt COCO
df_coco = {}
df_coco['images'] = df_img.to_dict('records')
df_coco['annotations'] = df_obj.to_dict('records')
df_coco['categories'] = [{'id': v, 'name': k,
                          'supercategory': 'label'} for k, v in cats_id.items()]

with open('../dataset/coco.json',  "w") as json_file:
    json.dump(df_coco, json_file, indent=4, cls=NpEncoder)

import os 
import matplotlib.pyplot as plt
import seaborn 
import cv2
import numpy as np 
import os.path as osp
import flyr
from pycocotools.coco import COCO
import sys
sys.path.append('/home/user304/users/jiwon/fever_identification')
import utils
import utils.type_converter as tcvt
import pandas as pd
import json

def visual_flyr(img_prefix , num  ,get_original = False , get_mask = True  ):
    df = pd.read_csv('../dataset/img_metas.csv' )
    with open("../result/non_normalize_test/pred_result.json", "r") as st_json:
        result = json.load(st_json)
    plt.subplot(121)
    file_name = f'FLIR{str(num).zfill(4)}.jpg'
    file_path = osp.join(img_prefix , file_name)
    plt.title(f"truth_temp : {df[df['No.'] == num]['ear_temp'].values[0]}\n\
               pred_temp : {result[0][file_name]}")
    img =  cv2.imread(file_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if get_original:
        plt.subplot(122)
        thermogram = flyr.unpack(file_path)
        plt.imshow(thermogram.optical)
    if get_mask:
        plt.subplot(122)
        coco_path = '../dataset/coco.json'
        coco_cvt = utils.type_converter.COCO_converter(coco_gt_path = coco_path)
        img_id = coco_cvt.img_name_to_id[file_name]
        one_hot_mask = tcvt.get_one_hot_mask(coco_cvt.coco_gt , img_id)
        mask = tcvt.overlay(one_hot_mask)
        plt.imshow(mask)
        

    
import fiftyone.brain as fob
import json
import fiftyone as fo
from pycocotools.coco import COCO
import os
import re
import pandas as pd

''' fiftyone 을 활용한 mask 시각화를 위한 script'''

# get groundtruth info
coco = COCO('./dataset/val.json')
img_metas = pd.read_csv('./dataset/img_metas.csv')
img_prefix = './dataset/img/'
pattern = r'\d+'

preds_path = './result/non_normalize_test/pred_result.json'

if preds_path: 
    with open(preds_path, 'r') as f:
        pred_list = json.load(f)

def num_to_file_name(num):
    return f'FLIR{str(num).zfill(4)}.jpg'

def min_max(bbox, img_w, img_h):
    x, w = [x / img_w for x in bbox[::2]]
    y, h = [x / img_h for x in bbox[1::2]]
    return [x, y, w, h]


def mask_slice(bbox, mask):
    '''
    fiftyone은 bbox내 상대좌표로 masking을 함
    bbox 좌표 내로 자르기
    '''
    x1, y1, w, h = bbox
    return mask[y1:y1+h, x1:x1+w]


# coco dataset에 있는 image_path 불러오기
img_path_list = [anno['file_name'] for anno in coco.loadImgs(coco.getImgIds())]

# coco 데이터셋에서 카테고리 id와 이름을 매칭시키는 딕셔너리 생성
id_cat = {v['id']: v['name'] for v in coco.loadCats(coco.getCatIds())}

# 이미지 별로 fiftyone sample을 생성하여 detections 정보를 추가
samples = []
for img_id in coco.getImgIds():
    # fiftyone sample에 이미지 파일 경로를 추가
    img_info = coco.loadImgs(img_id)[0]
    sample = fo.Sample(filepath=img_prefix + img_info['file_name'])
    detections = []
    number =  int(re.findall(pattern , img_info['file_name'])[0])
    img_meta = img_metas[img_metas['No.'] == number]
    sample = fo.Sample(filepath=img_prefix + img_info['file_name'],
                         age = img_meta['age'].values[0],
                         sex = img_meta['sex'].values[0],
                         room_temp = img_meta['room_temp'].values[0],
                         ear_temp = img_meta['ear_temp'].values[0],
                         eye_temp = img_meta['eye_temp'].values[0],
                         skin_temp = img_meta['skin_temp'].values[0],
                         glabella_temp = img_meta['glabella_temp'].values[0],
                        pred_temp = pred_list[0][num_to_file_name(number)] ,
                         diff_temp = img_meta['ear_temp'].values[0] - pred_list[0][num_to_file_name(number)] )
    # 이미지에 대한 annotation 정보를 가져와서 detections에 추가
    for ann_id in coco.getAnnIds(imgIds=img_id):
        obj = coco.loadAnns(ann_id)[0]
        label = id_cat[obj['category_id']]  # annotation의 카테고리 id를 이용하여 라벨링
        # annotation의 bbox 정보를 이용하여 fiftyone detection의 bounding box 생성
        bbox = min_max(obj['bbox'], img_info['width'], img_info['height'])
        # fiftyone detection에 annotation 정보 추가
        
        detections.append(
            fo.Detection(label=label,
                         bounding_box=bbox,
                         mask=mask_slice(bbox=obj['bbox'], mask=coco.annToMask(obj)),
                         )
        )

    # fiftyone sample에 detections 정보 추가
    if detections:
        sample['ground_truth'] = fo.Detections(detections=detections)

    samples.append(sample)

# dataset title 넣기
dataset = fo.Dataset()
dataset.add_samples(samples)
dataset.add_sample_field( field_name='ground_truth.age', ftype = fo.core.fields.IntField , description ='An age')
dataset.add_sample_field( field_name='ground_truth.sex', ftype = fo.core.fields.StringField , description ='An sex')
dataset.add_sample_field( field_name='ground_truth.room_temp', ftype = fo.core.fields.FloatField , description ='An room_temp')
dataset.add_sample_field( field_name='ground_truth.ear_temp', ftype = fo.core.fields.FloatField , description ='An ear_temp')
dataset.add_sample_field( field_name='ground_truth.eye_temp', ftype = fo.core.fields.FloatField , description ='An eye_temp')
dataset.add_sample_field( field_name='ground_truth.skin_temp', ftype = fo.core.fields.FloatField , description ='An skin_temp')
dataset.add_sample_field( field_name='ground_truth.glabella_temp', ftype = fo.core.fields.FloatField , description ='An glabella_temp')
dataset.add_sample_field( field_name='ground_truth.pred_temp', ftype = fo.core.fields.FloatField , description ='An glabella_temp')
dataset.save()

if __name__ == "__main__":
    session = fo.launch_app(dataset, port=8820, address="0.0.0.0")
    session.wait()

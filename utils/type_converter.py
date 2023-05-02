import os
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np 
import pycocotools.mask as mask_cvt
import json
from torchvision.transforms import transforms

class NpEncoder(json.JSONEncoder):
    '''
    numpy to json format
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj , bytes):
            return obj.decode()
        return super(NpEncoder, self).default(obj)
    
def overlay(one_hot_mask):
    ''' (H,W,C) to overlay (H,W)
    Args: 
        one_hot_mask : (H,W,C)
    '''
    cmap = [(255,0,255) , (255,0,0) , (255,255,255) ]
    ch_c = one_hot_mask.shape[-1]
    mask = np.zeros((*one_hot_mask.shape[:2],3))
    for c in range(3):
        mask[np.where(one_hot_mask[:,:,c] != 0)] = cmap[c]
    return mask

def get_one_hot_mask( coco , idx):
    '''coco idx에 따른 one hot mask return
    '''

    catIds = len(coco.getCatIds())
    mask_list = []
    for i in range(1,catIds+1):
        mask_annos = coco.loadAnns(coco.getAnnIds(imgIds = idx, catIds = i))
        if len(mask_annos) > 1:
            mask = coco.annToMask(mask_annos[0])
            for mask_anno in mask_annos[1:]:
                mask_i = coco.annToMask(mask_anno)
                mask[mask_i > 0 ] = 1
        else:
            mask = coco.annToMask(mask_annos[0]) # 임시적인 mask
        mask_list.append(mask)
    return np.stack(mask_list , 2)

def get_celcius(img_name, path_prefix = './dataset/celsius/'):
    npy_name = img_name.replace('jpg' , 'npy')
    celsius_path= osp.join(path_prefix, npy_name)
    with open(celsius_path, 'rb') as f:
        celsius = np.load(f, encoding ='ASCII') 
    return celsius

class COCO_converter:
    ''' COCO path를 활용해 값을 도출하는 class

    '''

    def __init__(self, coco_gt_path, coco_pred_path=None):

        self.coco_gt = COCO(coco_gt_path)
        self.coco_pred = None

        if coco_pred_path:
            self.coco_pred = self.coco_gt.loadRes(coco_pred_path)

        self.img_name_to_id = {os.path.basename(image_info['file_name']): image_info['id'] for image_info in self.coco_gt.loadImgs(self.coco_gt.getImgIds())}
        self.imgId_to_name = {v: k for k, v in self.img_name_to_id.items()}
        self.cat_name_to_id = {cat_info['name']: cat_info['id'] for cat_info in self.coco_gt.loadCats(self.coco_gt.getCatIds())}
        self.catId_to_name = {v: k for k, v in self.cat_name_to_id.items()}
        self.cat_nums = len(self.coco_gt.getCatIds())



    
    def preds_to_json(self, img_metas , pred_masks , resize = True):
        if resize:
            trans = transforms.Resize(img_metas[0]['ori_img_size'][:2])
            pred_masks = trans(pred_masks)

        if not pred_masks.device.type == 'cpu': 
            pred_masks = pred_masks.cpu()
        if not isinstance(pred_masks , np.ndarray ):
            pred_masks = pred_masks.numpy().astype(np.uint8)
        return [{'image_id' : self.img_name_to_id[img_metas[pred_mask[0]]['file_name']],
                    'segmentation' : mask_cvt.encode(np.asfortranarray(pred_mask_i[1])),
                    'score' : 1.0,
                    'category_id' : pred_mask_i[0] } for pred_mask in enumerate(pred_masks) for  pred_mask_i in enumerate(pred_mask[1],1)]
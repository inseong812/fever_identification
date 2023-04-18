import torch
import numpy as np

def IoU(masks, preds, num_classes=1):
    
     # flatten
    masks = masks.view(-1)
    preds = preds.view(-1)
    
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    
    for i in range(num_classes):    #class하나이므로 굳이 for문x
        tp = ((preds==1) & (masks==1)).sum().float()
        fp = ((preds == 1) & (masks != 1)).sum().float()
        fn = ((preds != 1) & (masks == 1)).sum().float() 
        
        intersection[i] = tp
        union[i] = tp + fp + fn
    
    # 클래스 별 IoU 계산(현재는 class가 face 하나)
    iou = intersection / union
    
    # 평균 IoU 계산
    mean_iou = iou.mean().item()
    
    return mean_iou
    
    
        
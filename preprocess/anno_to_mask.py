from multiprocessing import Pool, cpu_count
from time import time
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import os
from functools import partial
from PIL import Image


import os
from multiprocessing import Pool
from functools import partial
from pycocotools.coco import COCO
from PIL import Image


def savefig(img_ids, coco):
    '''
    COCO 데이터셋에서 이미지의 마스크를 추출하여 저장하는 함수
    '''
    for img_id in img_ids:
        # 이미지 파일 이름 추출
        file_name = os.path.basename(coco.loadImgs(
            img_id)[0]['file_name']).split('.')[0]
        # 마스크 추출
        mask = coco.annToMask(coco.loadAnns(img_id)[0])
        # 추출한 마스크를 이미지로 변환하여 저장
        im = Image.fromarray(mask)
        im.save(f"../dataset/mask/{file_name}.png")


def parall_func(coco_path, func, num_cores):
    '''
    데이터를 병렬로 불러오기 위한 함수
    '''
    # 마스크를 저장할 디렉토리 생성
    os.makedirs('../dataset/mask/', exist_ok=True)
    # COCO 데이터셋 로드
    coco = COCO(coco_path)
    # 모든 이미지 ID 불러오기
    imgIds = coco.getImgIds()
    # 데이터를 나누기 위한 단위 계산
    data_lens = len(imgIds) // num_cores
    # savefig 함수에 coco 인자를 넘겨주기 위해 partial 함수 사용
    func = partial(func, coco=coco)
    # 이미지 ID 리스트를 데이터 단위에 맞게 나누기
    imgIds_list = [imgIds[i:i + data_lens]
                   for i in range(0, len(imgIds), data_lens)]

    print('프로세스 개수 :', num_cores)
    print('프로세스 당 데이터 개수 :', data_lens)
    # 데이터를 나누고, 코어 할당받기
    pool = Pool(num_cores + 1)
    pool.map(func, imgIds_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    n_cpu = cpu_count()
    print(f'n_cpu = {n_cpu}')
    parall_func(coco_path='../dataset/coco.json', func=savefig, num_cores=32)

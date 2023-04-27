import glob
import cv2
import tqdm
import numpy as np

''' image 별 pixel의 평균과 표준편차의 평균을 구하기 위한 script '''

img_path_list = glob.glob('./dataset/img/*')
print('image 개수 :', len(img_path_list))
mean_list = []
std_list = []
for img_path in tqdm.tqdm(img_path_list):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mean_list.append(img.mean())
    std_list.append(img.std())

print(np.array(mean_list) / len(mean_list) )
print(np.array(std_list) / len(std_list))

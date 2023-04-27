cd ../preprocess 
python preprocess/preprocess.py
python preprocess/FLIR_to_celsius.py
python preprocess/cocosplit.py --coco_path './dataset/coco.json' --train_size 0.8 --save_path './dataset/'
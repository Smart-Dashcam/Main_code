import torch
import ultralytics
from ultralytics import YOLO
#
# import os
# os.environ["OMP_NUM_THREADS"] = '1'

# ultralytics.checks()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

if __name__ == '__main__':
    # 모델 불러오기
    model = YOLO('yolov8n.pt')

    # 모델 학습하기
    model.train(data='D:/yolo_dataset2/data.yaml', epochs=100, patience=20, batch=16, imgsz=1920)
    # model.train(data='D:/yolo_dataset2/data.yaml', epochs=100, patience=20, batch=4, imgsz=1920) # yolov8m.pt
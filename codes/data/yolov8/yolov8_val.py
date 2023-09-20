from ultralytics import YOLO
import ultralytics
import torch
import pickle

ultralytics.checks()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('cpu')

if __name__ == '__main__':

    # Load our custom model
    model = YOLO('./runs/detect/model6  ASDz 2/weights/best.pt')
    # Use the model to detect object
    # metrics = model.val(data='D:/group7/yolo_dataset/dataset_v4/data.yaml', imgsz=1920, batch=8, plots=True, split='test')  # model4
    metrics = model.val(data='D:/yolo_dataset2/data.yaml', imgsz=1920, batch=4, plots=True, split='test')  # model5
    # metrics = model.val(data='D:/yolo_dataset/dataset_v4/data.yaml', imgsz=1920, batch=8, plots=True, split='test')  # model6

    with open(str(metrics.save_dir) + '/summary.pkl', 'wb') as f:
        pickle.dump(metrics.results_dict, f)




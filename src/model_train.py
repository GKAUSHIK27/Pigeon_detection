
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
 
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

#update the .yaml file path accordingly
model.train(data='//wsl.localhost/Ubuntu/home/govind/Development/data/YOLODataset_seg/dataset.yaml', epochs=100,imgsz=640 ,lr0=0.01,optimizer='Adam')








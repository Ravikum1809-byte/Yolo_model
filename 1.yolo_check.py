#pip install ultralytics
import ultralytics
ultralytics.checks() #check if everything is working fine
from ultralytics import YOLO
model = YOLO('yolov8n.pt') #load a pretrained model (nano version)

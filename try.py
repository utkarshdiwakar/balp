import cv2
import pandas
import numpy as np
from ultralytics import YOLO 
from enum import Enum
from ultralytics.utils.plotting import Annotator

class ModelType(Enum):
    YOLOv8n = 'yolov8n.pt'
    YOLOv8s = 'yolov8s.pt'
    YOLOv8x = 'yolov8x.pt'


def detection(modelType: ModelType, image_path: str):
    model = YOLO(modelType.value)
    results = model.predict(source=image_path, show=True)
    objects = {}
    for result in results:
        for box in result.boxes:
            left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
            h_mid = right - left
            height = bottom - top
            center = (left + int((right-left)/2), top + int((bottom-top)/2))
            label = results[0].names[int(box.cls)]
            confidence = float(box.conf.cpu())
            objects[label] = h_mid
    print(objects)

        

if __name__ == '__main__':
    image_path = 'snapshot.jpg'  # Provide the path to your image here
    detection(ModelType.YOLOv8s, image_path)


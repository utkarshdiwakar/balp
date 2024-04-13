"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from ultralytics import YOLO 
from enum import Enum
import time

# Initialize variables to track time
start_time = None
looking_direction = None
def capncrop(frame):
    
    class ModelType(Enum):
        YOLOv8n = 'yolov8n.pt'
        YOLOv8s = 'yolov8s.pt'
        YOLOv8x = 'yolov8x.pt'


    def detection(modelType: ModelType, image_path: str):
        model = YOLO(modelType.value)
        results = model.predict(source=image_path, show=True)
        objects = []
        print(results)
        for detection in results:
            class_id = detection['class_id']
            class_name = model.get_class_name(class_id)
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']
            x_center = x + w / 2
            y_center = y + h / 2
            objects.append({'class_name': class_name, 'confidence': confidence,
                            'x_center': x_center, 'y_center': y_center})
        print(objects)
            

    if __name__ == '__main__':
        image_path = 'snapshot.jpg'  # Provide the path to your image here
        detection(ModelType.YOLOv8s, image_path)

gaze = GazeTracking()
webcam = cv2.VideoCapture(1)
camera  = cv2.VideoCapture(0)

def take_snapshot(camera_index=0, save_path='snapshot.jpg'):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture frame.")
        return
    cv2.imwrite(save_path, frame)
    print(f"Snapshot saved as {save_path}")

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    __,frame2 = camera.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_right():
        if looking_direction != "right":
            looking_direction = "right"
            start_time = time.time()
        elif time.time() - start_time > 2:
            text = "Looking right"
            take_snapshot(camera_index=0, save_path='snapshot.jpg')
            break
    elif gaze.is_left():
        if looking_direction != "left":
            looking_direction = "left"
            start_time = time.time()
        elif time.time() - start_time > 2:
            text = "Looking Left"
            take_snapshot(camera_index=0, save_path='snapshot.jpg')
            break
    
    elif gaze.is_center():
        text = "Looking center"
    print(text)

    # image = cv2.imread('image')
    # gaze.refresh(image)
    # frame = gaze.annotated_frame()

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)
    cv2.imshow('s2', frame2)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
capncrop(frame2)
cv2.destroyAllWindows()

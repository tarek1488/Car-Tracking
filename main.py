import os
import cv2
from ultralytics import YOLO
import random
from tracker import Tracker 
import numpy as np
import easyocr
from utils import *
video_data = {}
input_video =  r'input_videos\in6.mp4'

capture = cv2.VideoCapture(input_video)

#loading needed models
car_detector =  YOLO('yolov8n.pt')
plate_detector = YOLO(r'plate-detector\train-results\weights\plate-detector.pt')

#loading deepsort tracker
tracker =  Tracker()

#Intialize OCR reader
reader = easyocr.Reader(['en'], gpu=False) 

#generate 10 random colors for boxes
color = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(10)]

frame_num = -1
#looping through video frames
while True:
    frame_num += 1
    ret, frame = capture.read()
    
    #detect cars in frames
    car_results = car_detector(frame,classes = 2)
    
    plate_results = plate_detector(frame)
    
    car_detections = []
    plate_detections = []
    #looping through results for each frame
    for r in car_results[0].boxes.data.tolist():
        #gathering data for tracking
        x1, y1, x2, y2, score, class_id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        car_detections.append([x1, y1, x2, y2, score])    
    
    #update tracker to make tracker track cars in the whole video
    tracker.update(frame, np.array(car_detections))
    
    for r in plate_results[0].boxes.data.tolist():
        #gathering data for tracking
        x1, y1, x2, y2, score, class_id = r
        #assign plate to car
        car_data =  assign_plate_to_car(r[0:4], tracker.tracks)
        
        if car_data != -1:
            #cropping plates
            plate = frame[int(y1):int(y2), int(x1):int(x2), :]
            x1_car, y1_car, x2_car, y2_car = car_data.bbox
            #converting plate to grayscale
            gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            #reading text from plate
            plate_text = reader.readtext(gray_plate, detail=0)
            if len(plate_text) != 0:
                video_data[frame_num][car_data.track_id] = {'car' : {'bbox': [int(x1_car), int(y1_car), int(x2_car), int(y2_car)]},
                                                            'licence plate': {'bbox':[int(x1), int(y1), int(x2), int(y2)],
                                                                              'text': plate_text[0]}}
    if not ret:
        print(f'Video {input_video} has ended')
        break
    
# Release resources
capture.release()
cv2.destroyAllWindows()
print(video_data)
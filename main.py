import os
import cv2
from ultralytics import YOLO
import random
from tracker import Tracker 
import numpy as np
import easyocr

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

#looping through video frames
while True:
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
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color[2], 3)
        #plate_detections.append([x1, y1, x2, y2])
        plate = frame[y1:y2, x1:x2, :]
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate_text = reader.readtext(plate, detail=0)
        if len(plate_text) != 0:
            txt = plate_text[0]
            with open('out.txt', 'a') as file:
                file.write(txt + "\n")
        
        
        
    
    if not ret:
        print(f'Video {input_video} has ended')
        break
    
    #cv2.imshow('frame',frame)
    cv2.waitKey(10)
    
    # key = cv2.waitKey(0)
    
    # if key == ord('q'):
    #     # 'q' to quit
    #     break
    # elif key == ord('n'):
    #     # 'n' to show the next frame
    #     continue
    # elif key == ord('s'):
    #     # 's' to save the current frame
    #     cv2.imwrite('saved_frame.png', frame)
    #     print("Frame saved as 'saved_frame.png'.")
    
    #cv2.waitKey(int(capture.get(cv2.CAP_PROP_FPS)))
    
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     print("Video processing interrupted by user.")
    #     break

# Release resources
capture.release()
cv2.destroyAllWindows()

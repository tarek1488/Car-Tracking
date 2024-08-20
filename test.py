import cv2
from ultralytics import YOLO
from easyocr import Reader

plate_detector = YOLO(r'plate-detector\train-results\weights\plate-detector.pt')

reader = Reader(['en'])


image = cv2.imread(r'C:\Users\Tarek\fiftyone\open-images-v7\train\data\0a699c9b1dafc8e0.jpg')
results = plate_detector(image)[0].boxes.data.tolist()[0]

x1, y1, x2, y2 = results[0:4]
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
image = image[y1:y2, x1:x2, :]


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


result = reader.readtext(gray_image)
print(result)
cv2.imwrite('plate-test.jpg',image)
cv2.imshow('frame',gray_image)

cv2.waitKey(0)
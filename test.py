from ultralytics import YOLO
import cv2
import easyocr
import ssl
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8_2/weights/best.pt")

im2 = cv2.imread("test_4.jpg")
results = model.predict(source=im2, save=False)

def preprocess(img):

    img = im2[y1:y2, x1:x2]
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

for result in results:
    coordinates = result.boxes.xyxy

    for coord in coordinates:
        x1, y1, x2, y2 = map(int, coord)
        cv2.rectangle(im2, (x1, y1), (x2, y2), (0, 255, 255), 2)

image = preprocess(im2)

ocr = PaddleOCR(lang='en')
result = ocr.ocr(image, cls=False)
    
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[0])

cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
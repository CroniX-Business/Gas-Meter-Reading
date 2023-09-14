from ultralytics import YOLO
import cv2
import easyocr
import ssl
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8/weights/best.pt")

im = cv2.imread("test_4.jpg")
img = cv2.resize(im, (640,640))
results = model.predict(source=img, save=False)

def preprocess(img):

    img = img[y1:y2, x1:x2]
    #img = cv2.resize(img, (320,240))
    #norm_img = np.zeros((img.shape[0], img.shape[1]))
    #img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

for result in results:
    coordinates = result.boxes.xyxy

    for coord in coordinates:
        x1, y1, x2, y2 = map(int, coord)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

image = preprocess(img)

ocr = PaddleOCR(lang='en')
result = ocr.ocr(image, cls=False)

'''    
for idx in range(len(result)):
    res = result[idx]
    print(res)
    for line in res:
        print(line[0])
'''

concatenated_result = ""

for idx in range(len(result)):
    res = result[idx]
    if isinstance(res, list) and len(res) > 1 and isinstance(res[1], tuple):
        extracted_value = res[1][0]
        concatenated_result += extracted_value

print(concatenated_result)



cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
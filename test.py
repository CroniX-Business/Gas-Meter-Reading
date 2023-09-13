from ultralytics import YOLO
import cv2
import easyocr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8/weights/best.pt")

im2 = cv2.imread("datasets/Brojilo/test/images/00882400012550_0_jpg.rf.8c4ca7d251ff58546adabc1173770ae5.jpg")
results = model.predict(source=im2, save=False)

for result in results:
    coordinates = result.boxes.xyxy

    for coord in coordinates:
        x1, y1, x2, y2 = map(int, coord)
        cv2.rectangle(im2, (x1, y1), (x2, y2), (0, 255, 255), 2)

for coord in coordinates:
    x1, y1, x2, y2 = map(int, coord)
    cropped_image = im2[y1:y2, x1:x2]
    upscaled_region = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(upscaled_region, detail=0, allowlist='0123456789')
    
    for res in result:
        print(res)
    
    #print("Text from cropped region:", result)

cv2.imshow('Image with Bounding Boxes', upscaled_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
for result in results:
   print(result.boxes.xyxy)

coordinates = result.boxes.xyxy

#x1, y1, x2, y2 = map(int, coordinates[0])

gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
cropped_image = gray[y1:y2, x1:x2]

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image, detail = 0, allowlist='0123456789')
print(result)

cv2.imshow('Cropped Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
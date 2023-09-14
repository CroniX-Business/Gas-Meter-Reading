import streamlit as st
from ultralytics import YOLO
#import easyocr
from paddleocr import PaddleOCR
import ssl
import cv2
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8/weights/best.pt")

def main():
    style()

    uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")

        image = cv2.resize(opencv_image, (640,640))
        results = model.predict(source=image, save=False)

        for result in results:
            coordinates = result.boxes.xyxy

            x1, y1, x2, y2 = map(int, coordinates[0])

        cropped_image = image[y1:y2, x1:x2]

        ocr = PaddleOCR(lang='en')
        result = ocr.ocr(cropped_image, cls=False)

        st.image(image, caption="Brojilo", use_column_width=True, channels="BGR")


        concatenated_result = ""

        for idx in range(len(result)):
            res = result[idx]
            if isinstance(res, list) and len(res) > 1 and isinstance(res[1], tuple):
                extracted_value = res[1][0]
                concatenated_result += extracted_value

        st.write(concatenated_result)


def style():
  st.set_page_config(page_title='Čitač plina')
  st.title("Očitavanje brojila plina")


if __name__ == '__main__':
  main()



#https://ocitavanje-mjerila-projekat.streamlit.app 
#Aplikacija radi preko github repository-a

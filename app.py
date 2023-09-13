import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
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

        results = model.predict(source=opencv_image, save=False)

    for result in results:
        coordinates = result.boxes.xyxy

        x1, y1, x2, y2 = map(int, coordinates)

        cropped_image = opencv_image[y1:y2, x1:x2]
        upscaled_region = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image, detail = 0, allowlist='0123456789')
        #print(result)

        st.image(upscaled_region, caption="Brojilo", use_column_width=True)
        st.write(result)


def style():
  st.set_page_config(page_title='Čitač plina')
  st.title("Očitavanje brojila plina")


if __name__ == '__main__':
  main()
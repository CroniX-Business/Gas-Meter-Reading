import streamlit as st
from ultralytics import YOLO
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
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR")

        results = model.predict(source=image, save=False)

        for result in results:
            coordinates = result.boxes.xyxy
            
            try:
                x1, y1, x2, y2 = map(int, coordinates[0])
            except IndexError:
                st.write("Brojilo nije prepoznato. Molimo uslikajte bolje te probajte da budete u ravnini")
                break
                
            cropped_image = image[y1:y2, x1:x2]
            img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)
            result = ocr.ocr(img, cls=False)
            print(result)
            
            st.image(cropped_image, caption="Brojilo", use_column_width=True, channels="BGR")
            
            concatenated_result = ""
            
            for idx in range(len(result)):
                res = result[idx]
                if isinstance(res, list) and len(res) > 1 and isinstance(res[1], tuple):
                    extracted_value = res[1][0]
                    if "m3" in extracted_value or "m" in extracted_value or extracted_value.isnumeric():
                        concatenated_result += extracted_value
                        print(concatenated_result)

            st.write(concatenated_result.replace("m3", "").replace("m", ""))

def style():
  st.set_page_config(page_title='Čitač plina')
  st.title("Očitavanje brojila plina")


if __name__ == '__main__':
  main()

#https://ocitavanjemjerilaprojekt.streamlit.app
#Aplikacija radi preko github repository-a

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
import ssl
import cv2
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8/weights/best.pt")


def main():
  style()

  uploaded_file = st.file_uploader("Odaberite sliku iz galerije",
                                   type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Originalna slika", use_column_width=True)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(source=image, save=False)

    for result in results:
        coordinates = result.boxes.xyxy

        x1, y1, x2, y2 = map(int, coordinates[0])

        cropped_image = image[y1:y2, x1:x2]
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
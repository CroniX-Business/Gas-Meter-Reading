import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO("runs/detect/yolov8/weights/best.pt")


def main():
  style()

  uploaded_file = st.file_uploader("Odaberite sliku iz galerije",
                                   type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Originalna slika", use_column_width=True)

    results = model.predict(source=image, save=False)

    for result in results:
        coordinates = result.boxes.xyxy

        x1, y1, x2, y2 = map(int, coordinates[0])

        #gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        cropped_image = image[y1:y2, x1:x2]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image, detail = 0, allowlist='0123456789')
        #print(result)

        st.write(result)

#cv2.imshow('Cropped Image', cropped_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


def style():
  st.set_page_config(page_title='Čitač plina')
  st.title("Očitavanje brojila plina")


if __name__ == '__main__':
  main()
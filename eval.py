from ultralytics import YOLO
 
model = YOLO('runs/detect/yolov8_2/weights/best.pt')
 
results = model.val(
   data='model.yaml',
   imgsz=640,
   name='yolov8n_2_eval'
)
from ultralytics import YOLO
 
model = YOLO('yolov8n.pt')
 
results = model.train(
   data='model.yaml',
   imgsz=640,
   epochs=20,
   batch=8,
   name='yolov8_2'
)
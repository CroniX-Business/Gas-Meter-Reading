from ultralytics import YOLO
 
model = YOLO('yolov8n.pt')
 
results = model.train(
   data='model.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   name='yolov8'
)
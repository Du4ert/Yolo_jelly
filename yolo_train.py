from ultralytics import YOLO
# from clearml import Task

# task = Task.init(project_name="my project", task_name="my task")
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=150,
   batch=8,
   name='yolo8n_v3'
   resume=True
)
from ultralytics import YOLO
# from clearml import Task

# task = Task.init(project_name="my project", task_name="my task")
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=100,
   batch=8,
   name='delete'
)
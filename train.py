from ultralytics import YOLO
import mlflow
# dataset
# https://universe.roboflow.com/mohamed-khaled-a0klw/plant-detect/dataset/2

mlflow.autolog()

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='./plant_detect/data.yaml', epochs=100, imgsz=640)
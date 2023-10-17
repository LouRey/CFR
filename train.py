from ultralytics import YOLO
import mlflow

def main():
    mlflow.autolog()

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='./plant_detect/data.yaml', epochs=100, imgsz=640)

if __name__ == '__main__':
    main()

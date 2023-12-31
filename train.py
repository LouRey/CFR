from ultralytics import YOLO
import mlflow

#to use model, get it from luxonis tool at http://tools.luxonis.com/
def main():
    mlflow.autolog()

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='./plant_detect/data.yaml', epochs=150, imgsz=640)

if __name__ == '__main__':
    main()

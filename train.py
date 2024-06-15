from ultralytics import YOLO

# Load a model
model = YOLO("pretrain/yolov8m.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model
    results = model.train(data="cfg/data/dataset.yaml", epochs=100, imgsz=640, batch=8)
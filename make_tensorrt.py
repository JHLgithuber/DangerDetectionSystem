from ultralytics import YOLO

model = YOLO("yolo11x-pose.pt")
model.export(format="engine", dynamic=True)  # creates 'yolo11n.engine'
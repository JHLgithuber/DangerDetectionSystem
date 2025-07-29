import cv2

from ultralytics import YOLO


class YOLOWorldTRT:
    def __init__(self, engine_path):
        # Ultralytics YOLO API로 TensorRT engine 로드
        self.model = YOLO(engine_path)

    def infer(self, image, conf=0.7):
        results = self.model.predict(source=image, conf=conf, verbose=False)
        if results:
            return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[
                0].boxes.cls.cpu().numpy()
        else:
            return [], [], []


def run_yolo_and_track(image, yolox_model, conf=0.7):
    boxes, confs, class_ids = yolox_model.infer(image, conf=conf)

    for box, conf_score, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"ID:{int(cls_id)} {conf_score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

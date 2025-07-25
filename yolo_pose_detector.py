import time

import cv2
from multiprocessing import Process, current_process, Queue
from ultralytics import YOLO
import dataclass_for_StreamFrameInstance


import cv2

import cv2

def run_yolo_pose_process(model_path, input_q, output_q, conf=0.3, max_batch_size=10, debug=False):
    yolo_pose_process = Process(
        name=f"yolo_pose_worker",
        target=yolo_pose_worker,
        args=(input_q, output_q, model_path, conf, max_batch_size, debug,),
    )
    yolo_pose_process.daemon=True
    yolo_pose_process.start()
    return yolo_pose_process

def yolo_pose_worker(input_q, output_q, model_path, conf=0.3, max_batch_size=50, debug=False,):
    detector=YOLOPoseDetector(model_path=model_path, conf=conf)
    stream_frame_instance_list=list()
    frame_list=list()
    while True:
        if debug: print(f"[DEBUG] yolo_pose_worker LOOP")
        try:
            while not input_q.empty() and len(stream_frame_instance_list)<max_batch_size:
                stream_frame_instance = input_q.get()
                stream_frame_instance.sequence_perf_counter["yolo_pose_start"]=time.perf_counter()
                stream_frame_instance_list.append(stream_frame_instance)
                frame=dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance)
                frame_list.append(frame)

            results=detector.run_inference(frame_list)
            #if debug: print(results)

            for stream_frame_instance, result in zip(stream_frame_instance_list, results):
                #if debug:
                #    vis=result.plot()
                #    cv2.imshow(stream_frame_instance.stream_name+" yolo_pose_debug",vis)
                #    cv2.waitKey(1)
                #print("result.boxes.xyxy.cpu().numpy()",result.boxes.xyxy.cpu().numpy())
                #stream_frame_instance.human_detection_numpy=result.boxes.xyxy.cpu().numpy()
                #print("result.keypoints.xy.numpy()",result.keypoints.xy.cpu().numpy())
                #stream_frame_instance.pose_detection_numpy=result.keypoints.xy.numpy()
                stream_frame_instance.sequence_perf_counter["yolo_pose_end"]=time.perf_counter()
                output_q.put(stream_frame_instance)
            stream_frame_instance_list.clear()
            frame_list.clear()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in yolo_pose_worker: {e}")



class YOLOPoseDetector:
    def __init__(self, model_path="yolo11x-pose.pt", conf=0.3, debug=False, ):
        self.model_path = model_path
        self.conf = conf
        self.debug = debug

        self.model = YOLO(model_path)
        self.model.predict()
        self.model.fuse()

    def run_inference(self, frames):
        return self.model(frames, conf=self.conf)



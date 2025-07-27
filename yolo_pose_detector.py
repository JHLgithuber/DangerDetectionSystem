import time
from queue import Empty
import cv2
from multiprocessing import Process, current_process, Queue
from ultralytics import YOLO
import dataclass_for_StreamFrameInstance


import cv2

import cv2

def run_yolo_pose_process(model_path, input_q, output_q, conf=0.3, max_batch_size=50, worker_num=2, debug=False):
    yolo_pose_processes=list()
    for i in range(worker_num):
        yolo_pose_process = Process(
            name=f"yolo_pose_worker",
            target=yolo_pose_worker,
            args=(input_q, output_q, model_path, conf, max_batch_size, debug,),
        )
        yolo_pose_process.daemon=True
        yolo_pose_process.start()
        yolo_pose_processes.append(yolo_pose_process)
        if debug: print(f"[DEBUG] yolo_pose_process {i} started")
    return yolo_pose_processes

def yolo_pose_worker(input_q, output_q, model_path, conf, max_batch_size, debug, plot=False,):
    detector=YOLOPoseDetector(model_path=model_path, conf=conf)
    stream_frame_instance_list=list()
    frame_list=list()
    while True:
        stream_frame_instance_list.clear()
        frame_list.clear()
        if debug: print(f"[DEBUG] yolo_pose_worker LOOP")
        try:
            # 첫 프레임을 blocking하게 대기 (최대 0.1초)
            stream_frame_instance = input_q.get(timeout=0.1)
            batch_time = time.perf_counter()
            stream_frame_instance.sequence_perf_counter["yolo_pose_start"] = batch_time
            stream_frame_instance_list.append(stream_frame_instance)
            frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance)
            frame_list.append(frame)

            # 이후는 non-blocking
            while len(stream_frame_instance_list) < max_batch_size and time.perf_counter() - batch_time < 0.2:
                try:
                    stream_frame_instance = input_q.get_nowait()
                    stream_frame_instance.sequence_perf_counter["yolo_pose_start"] = time.perf_counter()
                    stream_frame_instance_list.append(stream_frame_instance)
                    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance)
                    frame_list.append(frame)
                except Empty:
                    break

            if frame_list:
                results = detector.run_inference(frame_list)
                for stream_frame_instance, result in zip(stream_frame_instance_list, results):
                    if plot:
                        vis = result.plot()
                        cv2.imshow(stream_frame_instance.stream_name + " yolo_pose_debug", vis)
                        cv2.waitKey(1)

                    stream_frame_instance.human_detection_numpy = result.boxes.xyxy.cpu().numpy()
                    stream_frame_instance.pose_detection_numpy = result.keypoints.xy.cpu().numpy()
                    stream_frame_instance.pose_detection_conf = result.keypoints.conf.cpu().numpy()
                    stream_frame_instance.sequence_perf_counter["yolo_pose_end"] = time.perf_counter()
                    output_q.put(stream_frame_instance)

            time.sleep(0.0001)

        except Empty:
            continue  # input_q에 아무것도 없으면 다음 루프

        except KeyboardInterrupt:
            print(f"[yolo_pose_worker] {current_process().name} is ended by KeyboardInterrupt")
            raise KeyboardInterrupt
        except Exception as e:
            print(f"[yolo_pose_worker ERROR] {e}")



class YOLOPoseDetector:
    def __init__(self, model_path="yolo11x-pose.pt", conf=0.3, debug=False, ):
        self.model_path = model_path
        self.conf = conf
        self.debug = debug

        self.model = YOLO(model_path)

    def run_inference(self, frames):
        return self.model(frames, conf=self.conf, half=True, stream=True)



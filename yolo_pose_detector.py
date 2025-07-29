import time
import traceback
from multiprocessing import Process, current_process, Queue
from queue import Empty

import cv2

import dataclass_for_StreamFrameInstance
from ultralytics import YOLO


def run_yolo_pose_process(model_path, input_q, output_q, conf=0.3, worker_num=2, debug=False):
    yolo_pose_processes = list()
    round_robin_input_queues = list()
    round_robin_output_queues = list()
    for i in range(worker_num):
        round_robin_input_queue = Queue(maxsize=worker_num)
        round_robin_output_queue = Queue(maxsize=worker_num)
        yolo_pose_process = Process(
            name=f"yolo_pose_worker",
            target=yolo_pose_worker,
            args=(round_robin_input_queue, round_robin_output_queue, model_path, conf, debug,),
        )
        yolo_pose_process.daemon = True
        yolo_pose_process.start()
        yolo_pose_processes.append(yolo_pose_process)
        round_robin_input_queues.append(round_robin_input_queue)
        round_robin_output_queues.append(round_robin_output_queue)
        if debug: print(f"[DEBUG] yolo_pose_process {i} started")

    yolo_pose_round_robin_process = Process(
        name=f"yolo_pose_round_robin_worker",
        target=yolo_pose_round_robin_worker,
        args=(input_q, output_q, round_robin_input_queues, round_robin_output_queues),
    )
    yolo_pose_round_robin_process.daemon = True
    yolo_pose_round_robin_process.start()
    if debug: print(f"[DEBUG] yolo_pose_round_robin_process started")
    return yolo_pose_processes, yolo_pose_round_robin_process


def yolo_pose_round_robin_worker(input_q, output_q, round_robin_input_queues, round_robin_output_queues):
    input_queue_index = 0
    output_queue_index = 0
    try:
        while True:
            if not input_q.empty():
                input_data = input_q.get()
                round_robin_input_queues[input_queue_index].put(input_data)
                input_queue_index = (input_queue_index + 1) % len(round_robin_input_queues)

            try:
                output_data = round_robin_output_queues[output_queue_index].get_nowait()
                output_q.put(output_data)
            except Empty:
                continue
            finally:
                output_queue_index = (output_queue_index + 1) % len(round_robin_output_queues)

    except KeyboardInterrupt:
        print(f"[yolo_pose_round_robin_worker] {current_process().name} is ended by KeyboardInterrupt")

    except Exception as e:
        print(f"[yolo_pose_round_robin_worker ERROR] {e}")
        traceback.print_exc()
        raise e


# noinspection PyTypeChecker
def yolo_pose_worker(input_q, output_q, model_path, conf, debug=False, plot=False, ):
    detector = YOLOPoseDetector(model_path=model_path, conf=conf)
    while True:
        try:
            # 첫 프레임을 blocking하게 대기 (최대 0.1초)
            stream_frame_instance = input_q.get()
            batch_time = time.perf_counter()
            stream_frame_instance.sequence_perf_counter["yolo_pose_start"] = batch_time
            # stream_frame_instance_list.append(stream_frame_instance)
            frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance)
            # frame_list.append(frame)

            results = detector.run_inference(frame)
            # for stream_frame_instance, result in zip(stream_frame_instance_list, results):
            result = next(results)
            if debug: print(f"[DEBUG] yolo_pose_worker {current_process().name} result.boxes.xyxy.shape={result.boxes.xyxy.shape}")
            if plot:
                vis = result.plot()
                cv2.imshow(stream_frame_instance.stream_name + " yolo_pose_debug", vis)
                cv2.waitKey(1)

            stream_frame_instance.human_detection_numpy = result.boxes.xyxy.cpu().numpy()
            stream_frame_instance.pose_detection_numpy = result.keypoints.xy.cpu().numpy()
            stream_frame_instance.pose_detection_conf = result.keypoints.conf.cpu().numpy()
            stream_frame_instance.sequence_perf_counter["yolo_pose_end"] = time.perf_counter()
            if output_q.full():
                output_q.get()
            output_q.put(stream_frame_instance)

            time.sleep(0.0001)


        except KeyboardInterrupt:
            print(f"[yolo_pose_worker] {current_process().name} is ended by KeyboardInterrupt")
        except Exception as e:
            print(f"[yolo_pose_worker ERROR] {e}")
            traceback.print_exc()


class YOLOPoseDetector:
    def __init__(self, model_path="yolo11x-pose.pt", conf=0.3, debug=False, ):
        self.model_path = model_path
        self.conf = conf
        self.debug = debug

        self.model = YOLO(model_path)

    def run_inference(self, frames):
        return self.model(frames, conf=self.conf, half=True, stream=True)

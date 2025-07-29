import time
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Queue
import cv2
import numpy as np
from stream_input import InputStream
import falling_iou_checker
import yolo_pose_detector
from demo_viewer import start_imshow_demo

def init_detect_flow(io_queue, cap):
    try:
        raw_cv2_frame_input_queue, classified_queue = io_queue
        for _ in range(120):
            ret, frame = cap.read()
            if ret:
                raw_cv2_frame_input_queue.put(frame)
            if not classified_queue.empty():
                classified_queue.get()
            time.sleep(0.001)

        while not raw_cv2_frame_input_queue.empty() or classified_queue.empty():
            classified_queue.get()
            print("queue is not empty")
    except Exception as e:
        print(f"init_detect_flow error: {e}")
        import traceback
        traceback.print_exc()


def simple_detect(io_queue, frame, pre_processed_frame=None):
    try:
        raw_cv2_frame_input_queue, classified_queue = io_queue
        while not classified_queue.empty():
            classified_queue.get_nowait()
            print("queue is not empty")
        raw_cv2_frame_input_queue.put(frame)
        print("raw_cv2_frame_input_queue.put is done")
        processed_frame = classified_queue.get(timeout=10.0)

        if pre_processed_frame is not None:
            processed_frame = np.where(processed_frame != 0, processed_frame, pre_processed_frame)

        return processed_frame
    except Exception as e:
        print(f"simple_detect error: {e}")
        import traceback
        traceback.print_exc()
        return frame


def output_stream_classifier(output_queue, classified_queues):
    while True:
        src, output_frame = output_queue.get()
        classified_queues[src].put(output_frame)
        time.sleep(0.0001)


def fall_detect_init(sources, max_frames=240, overlay_output=True, debug_mode=True):
    """
    :param shm_names_dict:
    :param overlay_output:
    :param sources: 스트림 주소나 cam_id
    :param max_frames: 스트림당 메모리 할당량
    :param debug_mode: 디버그 모드

    :return: 스트림별 입력큐, 출력큐
    """
    stream_many = len(sources)

    # 입력 스트림 초기화
    input_metadata_queue = Queue(maxsize=60 * stream_many)
    raw_cv2_frame_input_queues = dict()
    stream_instance_dict = dict()
    for i, src in enumerate(sources):
        str_src = str(src)
        print(f"name: {src}, url: {src}")

        cap = cv2.VideoCapture(src)
        raw_cv2_frame_input_queues[str_src] = Queue(maxsize=5)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                raise Exception("Stream initialization failed")
            raw_cv2_frame_input_queues[str_src].put(frame)
        cap.release()

        stream_instance_dict[str_src] = InputStream(source_path=raw_cv2_frame_input_queues[str_src],
                                                    metadata_queue=input_metadata_queue,
                                                    stream_name=str_src,
                                                    receive_frame=1, ignore_frame=0,
                                                    resize=None,
                                                    media_format="cv2_frame", debug=debug_mode)

    # 공유메모리 설정
    frame_smm_mgr = SharedMemoryManager()
    frame_smm_mgr.start()
    shm_objs_dict = dict()
    shm_names_dict = dict()
    for name, instance in stream_instance_dict.items():
        shm_objs = [frame_smm_mgr.SharedMemory(size=instance.get_bytes()) for _ in range(max_frames)]
        for shm in shm_objs: shm.buf[:] = b'\0' * instance.get_bytes()
        shm_name = [shm.name for shm in shm_objs]
        shm_objs_dict[name] = shm_objs
        shm_names_dict[name] = shm_name
        if debug_mode: print(f"shm_dict_name: {name}")

    classified_queues = dict()
    not_classified_queue = Queue(maxsize=5)
    for src in sources:
        classified_queues[str(src)] = Queue(maxsize=5)
    from threading import Thread
    classified_thread = Thread(target=output_stream_classifier,
                               args=(not_classified_queue, classified_queues))
    classified_thread.start()
    if debug_mode: print(f"classified_thread.is_alive: {classified_thread.is_alive()}")

    # 출력 스트림 설정
    output_metadata_queue = Queue(maxsize=30 * stream_many)
    # headless       => False: imshow 화면 전시, True: 로컬 화면 전시 없음
    # server_queue   => None: 웹 뷰어 사용안함, output_metadata_queue: 웹 뷰어 큐
    # visual         => True: 화면 합성, False: 화면 합성 없음(CLI Only)
    demo_process = start_imshow_demo(stream_queue=output_metadata_queue,
                                     server_queue=not_classified_queue,
                                     headless=True,
                                     show_latency=True, show_fps=True, visual=True,
                                     overlay=True, debug=debug_mode)
    if debug_mode: print(f"demo_process.is_alive: {demo_process.is_alive()}")

    # Pose Estimation
    after_pose_estimation_queue = Queue(maxsize=70 * stream_many)
    pose_processes, manager_process = yolo_pose_detector.run_yolo_pose_process(model_path="yolo11x-pose.engine",
                                                                               input_q=input_metadata_queue,
                                                                               output_q=after_pose_estimation_queue,
                                                                               conf=0.3,
                                                                               max_batch_size=20,
                                                                               worker_num=6,
                                                                               debug=debug_mode,
                                                                               )
    if debug_mode:
        for pose_process in pose_processes:
            print(f"pose_process.is_alive: {pose_process.is_alive()}")
        print(f"manager_process.is_alive: {manager_process.is_alive()}")

    # Falling multi frame IoU Checker
    fall_checker = falling_iou_checker.run_fall_worker(input_q=after_pose_estimation_queue,
                                                       output_q=output_metadata_queue,
                                                       buffer_size=50,
                                                       fall_ratio_thresh=0.7,
                                                       debug=debug_mode)
    if debug_mode: print(f"fall_checker.is_alive: {fall_checker.is_alive()}")

    # 입력 스트림 실행
    for name, instance in stream_instance_dict.items():
        instance.run_stream(shm_names_dict[name], )

    processes_dict = dict()
    processes_dict["demo_process"] = demo_process
    processes_dict["pose_processes"] = pose_processes
    processes_dict["fall_checker"] = fall_checker
    processes_dict["classified_thread"] = classified_thread
    processes_dict["stream_instance_dict"] = stream_instance_dict
    processes_dict["manager_process"] = manager_process

    io_queues = dict()
    for src in sources:
        io_queues[str(src)] = (raw_cv2_frame_input_queues[str(src)], classified_queues[str(src)])

    if debug_mode: print(f"init is done")
    return io_queues, frame_smm_mgr, shm_objs_dict, processes_dict

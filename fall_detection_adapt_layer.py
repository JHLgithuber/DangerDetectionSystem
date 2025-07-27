import time
from multiprocessing.managers import SharedMemoryManager

import cv2
import numpy as np

import falling_iou_checker
import yolo_pose_detector
from demo_viewer import start_imshow_demo


def simple_detect(io_queue, frame, pre_processed_frame=None):
    raw_cv2_frame_input_queue, classified_queue = io_queue
    while classified_queue.empty():
        classified_queue.get()
    raw_cv2_frame_input_queue.put(frame)
    processed_frame = classified_queue.get()

    if pre_processed_frame is not None:
        processed_frame = np.where(processed_frame != 0, processed_frame, pre_processed_frame)

    return processed_frame


def _output_stream_classifier(output_queue, classified_queues, sources):
    while True:
        src, output_frame = output_queue.get()
        classified_queues[src].put(output_frame)
        time.sleep(0.0001)


def fall_detect_init(sources, max_frames=500, overlay_output=True, debug_mode=True):
    """
    :param overlay_output:
    :param sources: 스트림 주소나 cam_id
    :param max_frames: 스트림당 메모리 할당량
    :param debug_mode: 디버그 모드

    :return: 스트림별 입력큐, 출력큐
    """
    frame_smm_mgr = SharedMemoryManager()
    frame_smm_mgr.start()
    stream_many = len(sources)

    # 입력 스트림 초기화
    from multiprocessing import Queue
    from stream_input import InputStream
    input_metadata_queue = Queue(maxsize=60 * stream_many)
    raw_cv2_frame_input_queues = dict()
    stream_instance_dict = dict()
    for i, src in enumerate(sources):
        src = str(src)
        print(f"name: {src}, url: {src}")
        raw_cv2_frame_input_queues[src] = Queue(maxsize=5)
        stream_instance_dict[src] = InputStream(source_path=raw_cv2_frame_input_queues[src],
                                                metadata_queue=input_metadata_queue,
                                                stream_name=str(src),
                                                receive_frame=1, ignore_frame=0,
                                                resize=None,
                                                media_format="cv2_frame", debug=debug_mode)
        cap = cv2.VideoCapture(src)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                raise Exception("Stream initialization failed")
            stream_instance_dict[src].put(frame)

    # 공유메모리 설정

    shm_objs_dict = dict()
    shm_names_dict = dict()
    for name, instance in stream_instance_dict.items():
        shm_objs = [frame_smm_mgr.SharedMemory(size=instance.get_bytes()) for _ in range(max_frames)]
        for shm in shm_objs: shm.buf[:] = b'\0' * instance.get_bytes()
        shm_name = [shm.name for shm in shm_objs]
        shm_objs_dict[name] = shm_objs
        shm_names_dict[name] = shm_name

    classified_queues = dict()
    not_classified_queue = Queue(maxsize=5)
    for src in sources:
        classified_queues[str(src)] = Queue(maxsize=5)
    from threading import Thread
    classified_thread = Thread(target=_output_stream_classifier,
                               args=(not_classified_queue, classified_queues, sources))

    # 출력 스트림 설정
    output_metadata_queue = Queue(maxsize=30 * stream_many)
    # headless       => False: imshow 화면 전시, True: 로컬 화면 전시 없음
    # server_queue   => None: 웹 뷰어 사용안함, output_metadata_queue: 웹 뷰어 큐
    # visual         => True: 화면 합성, False: 화면 합성 없음(CLI Only)
    demo_process = start_imshow_demo(stream_queue=output_metadata_queue,
                                     server_queue=not_classified_queue,
                                     headless=True,
                                     show_latency=True, show_fps=True, visual=True,
                                     overlay=overlay_output, debug=debug_mode)

    # Pose Estimation
    after_pose_estimation_queue = Queue(maxsize=70 * stream_many)
    pose_processes, manager_process = yolo_pose_detector.run_yolo_pose_process(model_path="yolo11x-pose.engine",
                                                                               input_q=input_metadata_queue,
                                                                               output_q=after_pose_estimation_queue,
                                                                               conf=0.3,
                                                                               max_batch_size=20,
                                                                               worker_num=6,
                                                                               debug=debug_mode, )

    # Falling multi frame IoU Checker
    fall_checker = falling_iou_checker.run_fall_worker(input_q=after_pose_estimation_queue,
                                                       output_q=output_metadata_queue,
                                                       buffer_size=50,
                                                       fall_ratio_thresh=0.7,
                                                       debug=debug_mode)

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

    return io_queues, processes_dict

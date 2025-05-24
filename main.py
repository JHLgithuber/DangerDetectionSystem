import argparse
import time
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Queue, freeze_support, set_start_method, cpu_count
from threading import Thread
from yolox.exp import get_exp
import cv2
import pose_detector
import dataclass_for_StreamFrameInstance
import human_detector
# from queue import Queue
from stream_input import RtspStream
from demo_viewer import start_imshow_demo
import sys


def get_args():
    hard_args = argparse.Namespace(
        demo="video",
        experiment_name=None,
        name="yolox-x",
        path="data_for_test/streetTestVideo4.mp4",
        camid=0,
        show_result=True,
        exp_file=None,
        ckpt="yolox_x.pth",
        device="gpu",
        conf=0.45,  # 신뢰도
        nms=0.65,  # 클수록 겹치는 바운딩박스 제거
        tsize=640,
        fp16=False,
        legacy=False,
        fuse=False,
        trt=False
    )
    return hard_args


def main(url_list, debug_mode=True, show_mode=True, show_latency=True, max_frames=1000):
    frame_smm_mgr = SharedMemoryManager()
    frame_smm_mgr.start()
    stream_instance_dict = dict()
    yolox_process = None
    mp_processes = None

    try:
        #프로세스 코어수 연동
        logical_cores = cpu_count()
        yolox_cores = max(int(logical_cores // 2.5),2)
        mp_cores = max(int((logical_cores - yolox_cores) // 1.2),2)
        print(f"logical_cores: {logical_cores}, yolox_cores: {yolox_cores}, mp_cores: {mp_cores}")


        stream_many=len(url_list)

        # 입력 스트림 초기화
        input_metadata_queue = Queue(maxsize=60*stream_many)
        for name, url, is_file in url_list:
            print(f"name: {name}, url: {url}")
            stream_instance_dict[name] = RtspStream(rtsp_url=url, metadata_queue=input_metadata_queue, stream_name=name,
                                                    receive_frame=1, ignore_frame=0,
                                                    startup_max_frame_count=200//logical_cores,
                                                    is_file=is_file, debug=debug_mode)
        print(f"stream_many: {stream_many}")

        # 공유메모리 설정
        shm_objs_dict = dict()
        shm_names_dict = dict()
        for name, instance in stream_instance_dict.items():
            shm_objs = [frame_smm_mgr.SharedMemory(size=instance.get_bytes()) for _ in range(max_frames)]
            for shm in shm_objs: shm.buf[:] = b'\0' * instance.get_bytes()
            shm_name = [shm.name for shm in shm_objs]
            shm_objs_dict[name] = shm_objs
            shm_names_dict[name] = shm_name
            # if debug_mode: print(shm_name)
            # if debug_mode: print(shm_objs)

        # 출력 스트림 설정
        output_metadata_queue = Queue(maxsize=3*stream_many)
        demo_thread = start_imshow_demo(output_metadata_queue, show_latency=show_latency, debug=debug_mode)

        # YOLOX ObjectDetection
        args = get_args()
        exp = get_exp(args.exp_file, args.name)
        after_object_detection_queue = Queue(maxsize=20*stream_many)
        yolox_process = human_detector.main(exp, args, input_metadata_queue, after_object_detection_queue,
                                            process_num=yolox_cores, all_object=False, debug_mode=debug_mode)
        yolox_process.start()

        # Sort before Tracking
        # TODO: 귀찮음

        # Pose Estimation
        mp_processes = pose_detector.run_pose_landmarker(process_num=mp_cores,
                                                         input_frame_instance_queue=after_object_detection_queue,
                                                         output_frame_instance_queue=output_metadata_queue,
                                                         debug=debug_mode, )

        # 입력 스트림 실행
        for name, instance in stream_instance_dict.items():
            instance.run_stream(shm_names_dict[name],)

        # 뭔가 프로세싱
        # while True:
        #    porc_frame=input_metadata_queue.get()
        #    #time.sleep(5)
        #    if debug_mode: print(f"porc_frame.captured_datetime: {porc_frame.captured_datetime}")
        #    output_metadata_queue.put(porc_frame)

        while True:
            time.sleep(2)
            # Bottle Neck Check
            if input_metadata_queue.full(): print("input_metadata_queue is FULL")
            if output_metadata_queue.full(): print("output_metadata_queue is FULL")
            if after_object_detection_queue.full(): print("after_object_detection_queue is FULL")

            #TODO: 메모리 오류유발
            #if (output_metadata_queue.empty() and
            #        not input_metadata_queue.empty() and
            #        not after_object_detection_queue.empty()):
            #    print("output_metadata_queue is EMPTY")
            #    for name, instance in stream_instance_dict.items():
            #        instance.startup_pass()
            #        if debug_mode: print(f"name: {name}, instance.startup_pass()")


            # Watch Dog
            if not demo_thread.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] demo thread is dead")
            if not yolox_process.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] yolox process is dead")
            if not all([mp_porc.is_alive() for mp_porc in mp_processes]):
                raise RuntimeError("[MAIN PROC WD ERROR] mp porc is dead")


    except KeyboardInterrupt:
        print("main end by KeyboardInterrupt")
    except RuntimeError as e:
        print(f"main RuntimeError: {e}")
    except Exception as e:
        print(f"main error: {e}")

    finally:
        exit_code=0
        try:
            # 리소스 정리
            if stream_instance_dict:
                for name, instance in stream_instance_dict.items():
                    thread=instance.kill_stream()
                    thread.join(timeout=5.0)
                    if debug_mode: print(f"name: {name}, instance.is_alive: {instance.is_alive()}")

            if yolox_process:
                yolox_process.terminate()
                yolox_process.join(timeout=5.0)

            if mp_processes:
                for mp_proc in mp_processes:
                    try:
                        mp_proc.terminate()
                        mp_proc.join(timeout=5.0)
                    except Exception as e:
                        print(f"프로세스 종료 중 오류: {e}")

            if frame_smm_mgr:
                frame_smm_mgr.shutdown()
                del frame_smm_mgr

        except Exception as e:
            print(f"정리 중 오류 발생: {e}")
            exit_code = 1

        return exit_code  # 종료 코드 반환


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    freeze_support()
    test_url_list = [
        # ("LocalHost", "rtsp://localhost:8554/stream"),
        ("TestFile_1", "data_for_test/streetTestVideo.mp4", True),
        ("TestFile_2", "data_for_test/streetTestVideo2.mp4", True),
        ("TestFile_3", "data_for_test/streetTestVideo3.mp4", True),
        ("TestFile_4", "data_for_test/streetTestVideo4.mp4", True),
        # ("CameraVidio","C:/Users/User/Pictures/Camera Roll/WIN_20250520_18_53_11_Pro.mp4",True),
        # ("TEST_0", "rtsp://210.99.70.120:1935/live/cctv068.stream", False),
        # ("TEST_1", "rtsp://210.99.70.120:1935/live/cctv069.stream", False),
        # ("TEST_2", "rtsp://210.99.70.120:1935/live/cctv070.stream", False),
        # ("TEST_3", "rtsp://210.99.70.120:1935/live/cctv071.stream", False),
        # ("TEST_4", "rtsp://210.99.70.120:1935/live/cctv072.stream", False),
        # ("TEST_5", "rtsp://210.99.70.120:1935/live/cctv073.stream", False),
        # ("TEST_6", "rtsp://210.99.70.120:1935/live/cctv074.stream", False),
        # ("TEST_7", "rtsp://210.99.70.120:1935/live/cctv075.stream", False),
        # ("TEST_8", "rtsp://210.99.70.120:1935/live/cctv076.stream", False),
        # ("TEST_9", "rtsp://210.99.70.120:1935/live/cctv077.stream", False),
    ]
    main(test_url_list)

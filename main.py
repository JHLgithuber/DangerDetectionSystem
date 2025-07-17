import argparse
import time
from multiprocessing import Queue, freeze_support, set_start_method, cpu_count
from multiprocessing.managers import SharedMemoryManager

import falling_iou_checker
import human_detector
import pose_detector
import stream_server as stream
from demo_viewer import start_imshow_demo
from stream_input import RtspStream
from yolox.exp import get_exp


def get_args():
    hard_args = argparse.Namespace(
        demo="video",
        experiment_name=None,
        name="yolox-x",
        path=None,
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


def main(url_list, debug_mode=False, show_latency=True, show_fps=True, print_visual=True, max_frames=1000):
    """
    전체 위험 감지 시스템 파이프라인 초기화 및 실행

    Args:
        url_list (list of tuple): (스트림 이름, URL, is_file 여부) 리스트
        debug_mode (bool): 디버그 출력 여부
        show_latency (bool): 프레임에 지연 시간 표시 여부
        show_fps (boot): 프레임에 fps 표시 여부
        max_frames (int): 스트림별 공유 메모리에 유지할 프레임 수
        print_visual: 결과 frame 합성 여부

    Returns:
        int: 종료 코드 (0: 정상 종료, 1: 예외 종료)

    Raises:
        RuntimeError: 주요 프로세스 중단 시 watchdog이 감지
        Exception: 설정 또는 실행 중 기타 예외 발생 가능
    """
    frame_smm_mgr = SharedMemoryManager()
    frame_smm_mgr.start()
    stream_instance_dict = dict()
    yolox_process = None
    mp_processes = None
    server_thread = None
    consumer_thread = None

    try:
        # 프로세스 코어수 연동
        logical_cores = cpu_count()
        yolox_cores = min(max(int(logical_cores // 3.5), 2), 6)
        mp_cores = max(int(logical_cores - yolox_cores - 5), 2)
        print(f"logical_cores: {logical_cores}, yolox_cores: {yolox_cores}, mp_cores: {mp_cores}")

        stream_many = len(url_list)

        # 입력 스트림 초기화
        input_metadata_queue = Queue(maxsize=60 * stream_many)
        for name, url, is_file in url_list:
            print(f"name: {name}, url: {url}")
            stream_instance_dict[name] = RtspStream(rtsp_url=url, metadata_queue=input_metadata_queue, stream_name=name,
                                                    receive_frame=1, ignore_frame=0,
                                                    startup_max_frame_count=int(200 / logical_cores),
                                                    resize=(854, 480),
                                                    #resize=None,
                                                    media_format=is_file, debug=debug_mode, startup_pass=False)
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

        # 스트리밍 서버 설정
        server_queue = Queue(maxsize=3 * stream_many)
        server_thread, consumer_thread= stream.run_stream_server(server_queue, host='0.0.0.0', port=5500)

        # 출력 스트림 설정
        output_metadata_queue = Queue(maxsize=3 * stream_many)
        #headless       => True: imshow 화면 전시, False: 로컬 화면 전시 없음
        #server_queue   => None: 웹 뷰어 사용안함, queue: 웹 뷰어 큐
        #visual         => True: 화면 합성, False: 화면 합성 없음(CLI Only)
        demo_thread = start_imshow_demo(stream_queue=output_metadata_queue, server_queue=None, headless=True,
                                        show_latency=show_latency, show_fps=show_fps, visual=print_visual, debug=debug_mode)

        # YOLOX ObjectDetection
        args = get_args()
        exp = get_exp(args.exp_file, args.name)
        after_object_detection_queue = Queue(maxsize=20 * stream_many)
        yolox_process = human_detector.main(exp, args, input_metadata_queue, after_object_detection_queue,
                                            process_num=yolox_cores, all_object=False, debug_mode=debug_mode)
        yolox_process.start()

        # Pose Estimation
        after_pose_estimation_queue = Queue(maxsize=20 * stream_many)
        mp_processes = pose_detector.run_pose_landmarker(process_num=mp_cores,
                                                         input_frame_instance_queue=after_object_detection_queue,
                                                         output_frame_instance_queue=after_pose_estimation_queue,
                                                         model_asset_path="pose_landmarker.task",
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

        while True:
            time.sleep(1)
            # Bottle Neck Check
            if input_metadata_queue.full(): print("[Warning] input_metadata_queue is FULL")
            if output_metadata_queue.full(): print("[Warning] output_metadata_queue is FULL")
            if after_object_detection_queue.full(): print("[Warning] after_object_detection_queue is FULL")
            if after_pose_estimation_queue.full(): print("[Warning] after_pose_estimation_queue is FULL")
            if server_queue.full(): print("[Warning] server_queue is FULL")

            # Watch Dog
            if not demo_thread.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] demo thread is dead")
            if not yolox_process.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] yolox process is dead")
            if not all([mp_porc.is_alive() for mp_porc in mp_processes]):
                raise RuntimeError("[MAIN PROC WD ERROR] mp porc is dead")
            if not fall_checker.is_alive():
                raise RuntimeError("[MAIN PROC WD ERROR] fall checker is dead")


    except KeyboardInterrupt:
        print("main end by KeyboardInterrupt")
    except RuntimeError as e:
        print(f"main RuntimeError: {e}")
    except Exception as e:
        print(f"main error: {e}")

    finally:  # 리소스 정리
        exit_code = 0
        try:
            if yolox_process:  # yolox 프로세스 종료
                yolox_process.terminate()
                yolox_process.join(timeout=5.0)

            if stream_instance_dict:  # 입력스트림 종료
                for name, instance in stream_instance_dict.items():
                    thread = instance.kill_stream()
                    thread.join(timeout=5.0)
                    print(f"name: {name}, instance.is_alive: {thread.is_alive()}")

            if mp_processes:  # 포즈 추정 프로세스 종료
                for mp_proc in mp_processes:
                    try:
                        mp_proc.terminate()
                        mp_proc.join(timeout=5.0)
                        print(f"mp_proc.is_alive: {mp_proc.is_alive()}")
                    except Exception as e:
                        print(f"프로세스 종료 중 오류: {e}")

            if frame_smm_mgr:  # 공유메모리 정리
                frame_smm_mgr.shutdown()
                del frame_smm_mgr

            print("See you later!")

        except Exception as e:
            print(f"정리 중 오류 발생: {e}")
            exit_code = 1

        finally:
            print(f"main END...{exit_code}")
    return exit_code


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    freeze_support()
    test_url_list = [
        # ("LocalHost", "rtsp://localhost:8554/stream"),
        # ("TestFile_1", "./data_for_test/streetTestVideo.mp4", "file"),
        # ("TestFile_2", "./data_for_test/streetTestVideo2.mp4", "file"),
        # ("TestFile_3", "./data_for_test/streetTestVideo3.mp4", "file"),
        # ("TestFile_4", "./data_for_test/streetTestVideo4.mp4", "file"),
        # ("Image_1", "./data_for_test/imageByCG.png", "file"),
        # ("Image_2", "data_for_test/ChatGPT Image 2025년 5월 19일 오전 12_49_16.png", "file"),
        # ("Image_3", "data_for_test/pose_demo_3p.png", "file"),
        # ("Image_4", "data_for_test/ChatGPT Image 2025년 5월 19일 오전 12_53_01.png", "file"),
        ("CameraVidio_1", "data_for_test/WIN_20250520_18_53_11_Pro.mp4", "file"),
        ("CameraVidio_2", "data_for_test/WIN_20250612_09_07_36_Pro.mp4", "file"),
        # ("LiveCamera_Windows", "video=Logitech BRIO", "dshow"),
        # ("SORA_1","data_for_test/CCTV_BY_CG_1.mp4","file"),
        # ("SORA_2","data_for_test/CCTV_BY_CG_2.mp4","file"),
        # ("SORA_3","data_for_test/CCTV_BY_CG_3.mp4","file"),
        # ("SORA_4","data_for_test/CCTV_BY_CG_4.mp4","file"),
        # ("TEST_0", "rtsp://210.99.70.120:1935/live/cctv068.stream", "rtsp"),
        # ("TEST_1", "rtsp://210.99.70.120:1935/live/cctv069.stream", "rtsp"),
        # ("TEST_2", "rtsp://210.99.70.120:1935/live/cctv070.stream", "rtsp"),
        # ("TEST_3", "rtsp://210.99.70.120:1935/live/cctv071.stream", "rtsp"),
        # ("TEST_4", "rtsp://210.99.70.120:1935/live/cctv072.stream", "rtsp"),
        # ("TEST_5", "rtsp://210.99.70.120:1935/live/cctv073.stream", "rtsp"),
        # ("TEST_6", "rtsp://210.99.70.120:1935/live/cctv074.stream", "rtsp"),
        # ("TEST_7", "rtsp://210.99.70.120:1935/live/cctv075.stream", "rtsp"),
        # ("TEST_8", "rtsp://210.99.70.120:1935/live/cctv076.stream", "rtsp"),
        # ("TEST_9", "rtsp://210.99.70.120:1935/live/cctv077.stream", "rtsp"),
    ]
    main(test_url_list)

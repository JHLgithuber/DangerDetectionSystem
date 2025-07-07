import time
from datetime import datetime
from multiprocessing import Process, Queue
from threading import Thread
import os
import cv2
import numpy as np
import torch
import platform
import dataclass_for_StreamFrameInstance
from pose_detector import crop_objects, draw_world_landmarks_with_coordinates
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import vis
import stream_server as stream

# noinspection PyUnusedLocal
def visual_from_fall_flag(stream_frame_instance,):
    """
    낙상 여부 플래그에 따라 포즈 오버레이 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 포즈 및 낙상 정보 포함 프레임 객체

    Returns:
        np.ndarray: 낙상 시각화가 적용된 프레임
    """
    # 1. 원본 프레임 로드
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, debug=True)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))

    # 2. 객체별 crop 정보 구하기
    crop_object_images = crop_objects(stream_frame_instance, need_frame=False)

    # 3. 각 객체(사람)별로 skeleton 그린 overlay로 합성
    for crop_object_img, pose_detection, fall_flag in zip(
            crop_object_images, stream_frame_instance.pose_detection_list, stream_frame_instance.fall_flag_list):

        # overlay에 skeleton 그리기
        pose_landmark_overlay = draw_world_landmarks_with_coordinates(
            pose_detection, img_size=crop_object_img["img_size"], )

        if fall_flag:
            cv2.putText(pose_landmark_overlay, "Triggered FALL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(pose_landmark_overlay, "NOT Triggered FALL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # bbox 좌표
        x1_p, y1_p, x2_p, y2_p = crop_object_img["bbox"]

        # 원본 frame의 해당 ROI 영역
        roi = frame[y1_p:y2_p, x1_p:x2_p]
        # 마스크: overlay 에서 검정이 아닌 부분만 True
        mask = np.any(pose_landmark_overlay != [0, 0, 0], axis=2)
        # ROI에 overlay: mask 부분만 복사
        roi[mask] = pose_landmark_overlay[mask]

    return frame


# noinspection PyUnusedLocal
def visual_from_pose_estimation(stream_frame_instance):
    """
    포즈 인식 결과에 대한 원본 프레임에 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 포즈 정보 포함 프레임 객체

    Returns:
        np.ndarray: 스켈레톤 오버레이가 적용된 프레임
    """
    # 1. 원본 프레임 로드
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, debug=True)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))

    # 2. 객체별 crop 정보 구하기
    crop_object_images = crop_objects(stream_frame_instance, need_frame=False)

    # 3. 각 객체(사람)별로 sk 그린 overlay 로 합성
    for crop_object_img, pose_detection in zip(
            crop_object_images, stream_frame_instance.pose_detection_list):
        # (1) crop 크기 만큼 검정 배경 생성
        # crop_h, crop_w = crop_object_img["crop"].shape[:2]
        # overlay = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        # (2) overlay에 skeleton 그리기
        pose_landmark_overlay = draw_world_landmarks_with_coordinates(
            pose_detection, img_size=crop_object_img["img_size"], )

        # (3) bbox 좌표
        x1_p, y1_p, x2_p, y2_p = crop_object_img["bbox"]

        # (4) 원본 frame의 해당 ROI 영역
        roi = frame[y1_p:y2_p, x1_p:x2_p]
        # (5) 마스크: overlay 에서 검정이 아닌 부분만 True
        mask = np.any(pose_landmark_overlay != [0, 0, 0], axis=2)
        # (6) ROI에 overlay: mask 부분만 복사
        roi[mask] = pose_landmark_overlay[mask]
        # (7) (frame은 numpy view 라서 자동 적용)

    return frame


def visual_from_detection_numpy(stream_frame_instance, cls_conf=0.35):
    """
    객체 탐지 결과를 프레임에 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 탐지 결과 포함 프레임 객체
        cls_conf (float): 클래스 confidence 필터 임계값

    Returns:
        np.ndarray: 시각화된 프레임
    """
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance, debug=True)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))
    test_size = (stream_frame_instance.human_detection_tsize, stream_frame_instance.human_detection_tsize)
    ratio = min(test_size[0] / frame.shape[0], test_size[1] / frame.shape[1])
    row_img = frame.copy()
    output = torch.tensor(stream_frame_instance.human_detection_numpy, dtype=torch.float32)

    if output is None:
        return row_img
    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(row_img, bboxes, scores, cls, cls_conf, COCO_CLASSES)  # frame에 결과 그려줌
    return vis_res





def demo_viewer(stream_name, frame, debug=False):
    """
    프레임을 화면에 표시

    Args:
        stream_name (str): 표시할 창 이름
        frame (np.ndarray): 표시할 영상 프레임
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[ERROR] {stream_name}: 유효 하지 않은 프레임")

        if frame.size == 0:
            print(f"[ERROR] {stream_name}: 빈 프레임")

        if debug: print(f"[DEBUG] {stream_name}: 프레임 shape: {frame.shape}, dtype: {frame.dtype}")
        cv2.imshow(stream_name, frame)

    except Exception as e:
        print(f"[ERROR] {stream_name} 뷰어 예외 발생: {e}")

def web_viewer(stream_name, frame, server_queue, debug=False):
    """
    frame을 웹에 표시

    Args:
        stream_name (str): 표시할 창 이름
        frame (np.ndarray): 표시할 영상 프레임
        server_queue : 서버 큐
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[web_viewer ERROR] {stream_name}: 유효 하지 않은 프레임")

        if frame.size == 0:
            print(f"[web_viewer ERROR] {stream_name}: 빈 프레임")

        if debug: print(f"[web_viewer DEBUG] {stream_name}: 프레임 shape: {frame.shape}, dtype: {frame.dtype}")
        server_queue.put((stream_name, frame))

    except Exception as e:
        print(f"[web_viewer ERROR] {stream_name} 뷰어 예외 발생: {e}")

def _add_latency_to_frame(frame, captured_datetime):
    """
    프레임에 지연 시간 텍스트 표시

    Args:
        frame (np.ndarray): 텍스트를 그릴 대상 프레임
        captured_datetime (datetime): 프레임 캡처 시각

    Returns:
        None
    """
    delta = datetime.now() - captured_datetime
    latency_s = int(delta.total_seconds())
    latency_us = int(delta.total_seconds() * 1_000_000)
    latency_text = f"Latency is {latency_us:08d} us, about {latency_s}seconds"

    cv2.putText(    # 윤곽선
        frame,
        latency_text,
        (20, 20),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale for better visibility
        (255, 255, 255),  # White color
        2,  # Thickness
        cv2.LINE_AA
    )

    # Add text to the top left of the frame
    cv2.putText(    # 지연시간
        frame,
        latency_text,
        (20, 20),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale for better visibility
        (0, 0, 0),  # Black color
        1,  # Thickness
        cv2.LINE_AA
    )


def _update_imshow_process(stream_queue_for_process, server_queue, headless=False, show_latency=False, debug=False):
    """
    스트림 프레임을 시각화하는 데모 프로세스
    큐에 들어온 데이터의 상태에 따라 조건문에 따른 적합한 시각화

    Args:
        stream_queue_for_process (Queue): StreamFrameInstance 객체가 담긴 큐
        show_latency (bool): 지연 시간 표시 여부
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """
    stream_name = stream_queue_for_process.get().stream_name
    print(f"[INFO] {stream_name} imshow demo process start")
    try:
        if not headless:
            cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(stream_name, 800, 600)

        sorter_gen = dataclass_for_StreamFrameInstance.sorter(messy_frame_instance_queue=stream_queue_for_process,
                                                              debug=debug)

        while True:
            instances_per_frame_instance = next(sorter_gen)
            if debug:
                print(f"[DEBUG] {stream_name} instances_per_frame_instance is {instances_per_frame_instance}")

            if instances_per_frame_instance is not None:
                if instances_per_frame_instance.fall_flag_list is not None: # 낙상 감지가 완료된 프레임
                    result_frame = visual_from_fall_flag(
                    stream_frame_instance=instances_per_frame_instance,
                )
                elif instances_per_frame_instance.pose_detection_list is not None:  #포즈 감지가 완료된 프레임
                    result_frame = visual_from_pose_estimation(
                        stream_frame_instance=instances_per_frame_instance,
                    )
                elif instances_per_frame_instance.human_detection_numpy is not None:    #객체 감지가 완료된 프레임
                    result_frame = visual_from_detection_numpy(
                        stream_frame_instance=instances_per_frame_instance,
                        cls_conf=0.35
                    )
                else:   # 별도의 처리가 없었던 프레임
                    result_frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
                        instances_per_frame_instance,
                        debug=debug
                    )

                # 지연 시간 추가
                if show_latency: _add_latency_to_frame(result_frame, instances_per_frame_instance.captured_datetime)



                if not headless: demo_viewer(stream_name, result_frame, debug=debug)
                if server_queue is not None:
                    if debug: print(f"[DEBUG] {stream_name} send to server")
                    web_viewer(stream_name, result_frame, server_queue, debug=debug)

            else:
                print(f"[INFO] {stream_name} instances_per_frame is None")
                break

            cv2.waitKey(1)
        cv2.destroyAllWindows()
    except Exception as e:
        cv2.destroyAllWindows()
        print(f"\n[ERROR] DEMO VIEWER of {stream_name} terminated due to: {e}")
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print(f"[INFO] DEMO VIEWER of {stream_name} ended by KeyboardInterrupt")


def _show_imshow_demo(stream_queue, server_queue, headless=False, show_latency=False, debug=False):
    """
    다중 스트림을 위한 imshow 데모 실행 및 프로세스 관리
    새로운 스트림이 들어오면 이에 대응하는 프로세스 생성

    Args:
        stream_queue (Queue): StreamFrameInstance 객체가 들어오는 메인 큐
        show_latency (bool): 지연 시간 표시 여부
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """
    stream_viewer_queue_dict = dict()
    stream_viewer_process_set = set()
    try:
        print("[INFO] imshow demo start")
        while True:
            stream = stream_queue.get()
            stream_name = stream.stream_name
            if stream_name not in stream_viewer_queue_dict:
                if debug: print(f"[DEBUG] {stream_name} is new in stream_viewer_queue_dict.")
                stream_viewer_queue_dict[stream_name] = Queue()
                process = Process(name=f"_update_imshow_process-{stream_name}", target=_update_imshow_process,
                                  args=(stream_viewer_queue_dict[stream_name], server_queue, headless, show_latency, debug))
                stream_viewer_process_set.add(process)
                process.start()
                time.sleep(0.001)
            stream_viewer_queue_dict[stream_name].put(stream)


    except KeyboardInterrupt:
        print("\nDEMO VIEWER is END by KeyboardInterrupt")

    except Exception as e:
        print(f"\nDEMO VIEWER is KILL by {e}")

    finally:
        def __terminate_process():
            for viewer in stream_viewer_process_set:
                viewer.terminate()
                viewer.join()
                print(f"[INFO] {viewer.name} is terminated.")
        __terminate_process()


def start_imshow_demo(stream_queue, server_queue=None, headless=False, show_latency=False, debug=False, ):
    """
    imshow 데모를 백그라운드 스레드로 시작

    Args:
        :param stream_queue: StreamFrameInstance 객체가 들어오는 메인 큐
        :param server_queue:
        :param debug: 디버그 메시지 출력 여부
        :param show_latency: 지연 시간 표시 여부
        :param headless: GUI창 미사용 여부

    Returns:
        Thread: 실행된 데모 스레드 객체

    """
    headless = headless or is_headless_cv2()

    imshow_demo_thread = Thread(name="_show_imshow_demo", target=_show_imshow_demo, args=(stream_queue, server_queue, headless, show_latency, debug))
    imshow_demo_thread.daemon = True
    imshow_demo_thread.start()
    return imshow_demo_thread

def is_headless_cv2():
    """
    안전하게 OpenCV GUI 기능이 비활성화된(headless) 환경인지 판단.
    - DISPLAY 환경변수로 X 서버 여부 확인 (Linux/macOS)
    - OpenCV 빌드 정보에서 GUI 백엔드 지원 여부 확인

    Returns:
        bool: True이면 headless 환경, False이면 GUI 사용 가능
    """
    # 1. DISPLAY 환경 변수 확인 (리눅스/macOS 한정)
    if platform.system() in ["Linux", "Darwin"]:
        if not os.environ.get("DISPLAY"):
            return True  # X 서버 없음

    # 2. OpenCV 빌드 정보에서 GUI 백엔드 지원 여부 확인
    try:
        info = cv2.getBuildInformation()
        gui_lines = [line for line in info.splitlines() if "GUI:" in line or any(g in line for g in ["GTK", "Qt", "Win32", "Cocoa", "Carbon"])]
        for line in gui_lines:
            if "YES" in line or "ON" in line:
                return False  # GUI 지원됨
        return True  # GUI 백엔드가 모두 NO인 경우
    except Exception:
        return True  # 정보 조회 실패 → 보수적으로 headless로 간주
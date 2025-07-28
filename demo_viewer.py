import os
import platform
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np

import dataclass_for_StreamFrameInstance


# noinspection PyUnusedLocal
def visual_from_fall_flag(stream_frame_instance, overlay=False, debug=True):
    """
    낙상 여부 플래그에 따라 포즈 오버레이 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 포즈 및 낙상 정보 포함 프레임 객체

    Returns:
        np.ndarray: 낙상 시각화가 적용된 프레임
        :param overlay:
        :param stream_frame_instance:
        :param debug:
    """
    # 1. 원본 프레임 로드
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, overlay=overlay, debug=debug)

    frame = _draw_bbox(frame, stream_frame_instance.human_detection_numpy)
    frame = _draw_skeleton(frame, stream_frame_instance.pose_detection_numpy)

    # 3. 각 객체(사람)별로 skeleton 그린 overlay로 합성
    for fall_flag, bbox in zip(stream_frame_instance.fall_flag_list, stream_frame_instance.human_detection_numpy):
        x1, y1, x2, y2 = map(int, bbox)
        if fall_flag:
            cv2.putText(frame, "Triggered FALL", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                        2)
        else:
            cv2.putText(frame, "NOT Triggered FALL", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 2)

    return frame


# noinspection PyUnusedLocal
def visual_from_pose_estimation(stream_frame_instance, overlay=False, debug=True):
    """
    포즈 인식 결과에 대한 원본 프레임에 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 포즈 정보 포함 프레임 객체

    Returns:
        np.ndarray: 스켈레톤 오버레이가 적용된 프레임
        :param overlay:
        :param stream_frame_instance:
        :param debug:
    """
    # 1. 원본 프레임 로드
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, overlay=overlay, debug=debug)

    frame = _draw_bbox(frame, stream_frame_instance.human_detection_numpy)
    keypoints = stream_frame_instance.pose_detection_numpy

    frame = _draw_skeleton(frame, keypoints)

    return frame


def _draw_skeleton(frame, keypoints, bone_color=(255, 255, 255), angle_color=(0, 255, 255)):
    # COCO keypoint 연결선 (관절 연결)
    coco_skeleton = [
        (5, 7), (7, 9),  # 왼팔
        (6, 8), (8, 10),  # 오른팔
        (11, 13), (13, 15),  # 왼다리
        (12, 14), (14, 16),  # 오른다리
        (5, 6),  # 어깨
        (11, 12),  # 엉덩이
        (5, 11), (6, 12),  # 몸통
    ]

    for keypoint in keypoints:
        # 점 그리기
        for (x, y) in keypoint:
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 2, angle_color, -1)

        # 선 그리기
        for j, k in coco_skeleton:
            x1, y1 = keypoint[j]
            x2, y2 = keypoint[k]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(frame, pt1, pt2, bone_color, 1)
    return frame


def visual_from_detection_numpy(stream_frame_instance, overlay=False, debug=True):
    """
    객체 탐지 결과를 프레임에 시각화

    Args:
        stream_frame_instance (StreamFrameInstance): 탐지 결과 포함 프레임 객체
        cls_conf (float): 클래스 confidence 필터 임계값

    Returns:
        np.ndarray: 시각화된 프레임
        :param overlay:
        :param stream_frame_instance:
        :param cls_conf:
        :param debug:
    """
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, overlay=overlay, debug=debug)
    boxes = stream_frame_instance.human_detection_numpy

    return _draw_bbox(frame, boxes)


def _draw_bbox(frame, boxes, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


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
        if server_queue.full():
            server_queue.get()
        server_queue.put((stream_name, frame))

    except Exception as e:
        print(f"[web_viewer ERROR] {stream_name} 뷰어 예외 발생: {e}")


def _add_latency_to_frame(captured_time, frame=None, debug=False):
    """
    프레임에 지연 시간 텍스트 표시

    Args:
        frame (np.ndarray): 텍스트를 그릴 대상 프레임
        captured_time (int): 프레임 캡처 시각(time.time_ns())

    Returns:
        None
    """
    delta_ns = time.time_ns() - captured_time
    latency_s = delta_ns // 1_000_000_000
    latency_text = f"Latency is {delta_ns:010d}ns, about {latency_s}seconds"
    if debug: print(latency_text)

    if frame is None:
        return latency_text

    cv2.putText(  # 윤곽선
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
    cv2.putText(  # 지연시간
        frame,
        latency_text,
        (20, 20),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale for better visibility
        (0, 0, 0),  # Black color
        1,  # Thickness
        cv2.LINE_AA
    )
    return latency_text


frame_id_time_list = list()


def _add_fps_to_frame(frame_id, frame=None, debug=False, ):
    """
    프레임에 FPS 표시

    Args:
        frame (np.ndarray): 텍스트를 그릴 대상 프레임

    Returns:
        None
    """
    global frame_id_time_list
    now = time.time()
    frame_id_time_list.append((frame_id, now))
    while True:
        if frame_id_time_list[0][1] < now - 1:
            frame_id_time_list.pop(0)
        else:
            break
    fps = len(frame_id_time_list)

    fps_text = f"FPS: {fps}"
    if debug: print(fps_text)

    if frame is None:
        return fps_text

    cv2.putText(  # 윤곽선
        frame,
        fps_text,
        (20, 50),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale for better visibility
        (255, 255, 255),  # White color
        2,  # Thickness
        cv2.LINE_AA
    )

    # Add text to the top left of the frame
    cv2.putText(  # FPS
        frame,
        fps_text,
        (20, 50),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale for better visibility
        (0, 0, 0),  # Black color
        1,  # Thickness
        cv2.LINE_AA
    )

    return fps


def _update_imshow_process(stream_queue_for_process, server_queue, headless=False, show_latency=False, show_fps=False,
                           visual=True, overlay=False, debug=False):
    """
    스트림 프레임을 시각화하는 데모 프로세스
    큐에 들어온 데이터의 상태에 따라 조건문에 따른 적합한 시각화

    Args:
        stream_queue_for_process (Queue): StreamFrameInstance 객체가 담긴 큐
        show_latency (bool): 지연 시간 표시 여부
        show_fps (bool): FPS 표시여부
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

        while True:
            instances_per_frame_instance = stream_queue_for_process.get()
            instances_per_frame_instance.sequence_perf_counter["viewer_before_visual"] = time.perf_counter()
            if debug:
                print(f"[DEBUG] {stream_name} instances_per_frame_instance is {instances_per_frame_instance}")

            if visual:
                if instances_per_frame_instance is not None:
                    if instances_per_frame_instance.fall_flag_list is not None:  # 낙상 감지가 완료된 프레임
                        result_frame = visual_from_fall_flag(
                            stream_frame_instance=instances_per_frame_instance,
                            overlay=overlay, debug=debug,
                        )
                    elif instances_per_frame_instance.pose_detection_numpy is not None:  # 포즈 감지가 완료된 프레임
                        result_frame = visual_from_pose_estimation(
                            stream_frame_instance=instances_per_frame_instance,
                            overlay=overlay, debug=debug,
                        )
                    elif instances_per_frame_instance.human_detection_numpy is not None:  # 객체 감지가 완료된 프레임
                        result_frame = visual_from_detection_numpy(
                            stream_frame_instance=instances_per_frame_instance,
                            overlay=overlay, debug=debug,
                        )
                    else:  # 별도의 처리가 없었던 프레임
                        result_frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
                            stream_frame_instance=instances_per_frame_instance,
                            overlay=overlay, debug=debug
                        )
                    instances_per_frame_instance.sequence_perf_counter["viewer_after_visual"] = time.perf_counter()

                    # 지연 시간 추가
                    if show_latency: _add_latency_to_frame(frame=result_frame,
                                                           captured_time=instances_per_frame_instance.captured_time,
                                                           debug=debug)
                    if show_fps: _add_fps_to_frame(frame=result_frame,
                                                   frame_id=instances_per_frame_instance.stream_name, debug=debug)

                    if not headless: demo_viewer(stream_name, result_frame, debug=debug)
                    if server_queue is not None:
                        if debug: print(f"[DEBUG] {stream_name} send to server")
                        web_viewer(stream_name, result_frame, server_queue, debug=debug)

            else:  # CLI Only 모드
                cli_text = f"[CLI_only] {stream_name} instances_per_frame\n"
                if show_fps: cli_text += f"[CLI_only] {stream_name} {_add_fps_to_frame(frame_id=instances_per_frame_instance.stream_name, debug=debug)}\n"
                if show_latency: cli_text += f"[CLI_only] {stream_name} {_add_latency_to_frame(captured_time=instances_per_frame_instance.captured_time, debug=debug)}\n"
                if instances_per_frame_instance.fall_flag_list is not None:
                    for fall_flag in instances_per_frame_instance.fall_flag_list:
                        cli_text += f"[CLI_only] {stream_name} fall_flag: {fall_flag}\n"
                else:
                    cli_text += f"[CLI_only] {stream_name} instances_per_frame is None"
                print(cli_text + "\n\n")

            if instances_per_frame_instance.sequence_perf_counter is not None:
                instances_per_frame_instance.sequence_perf_counter["demo_viewer_end"] = time.perf_counter()
                deltas = dataclass_for_StreamFrameInstance.compute_time_deltas(
                    instances_per_frame_instance.sequence_perf_counter)
                time_delta_log = f"[sequence time Delta INFO]\t{stream_name}\n"
                for step, delta in deltas.items():
                    time_delta_log += f"[sequence time Delta INFO]\t{step}\t:{delta:9.4f}ms\n"
                time_delta_log += f"[sequence time Delta INFO]\t{stream_name} OUT\n"
                print(time_delta_log)

            cv2.waitKey(1)
            time.sleep(0.02)
    except Exception as e:
        print(f"\n[ERROR] DEMO VIEWER of {stream_name} terminated due to: {e}")
    except KeyboardInterrupt:
        print(f"[INFO] DEMO VIEWER of {stream_name} ended by KeyboardInterrupt")
    finally:
        cv2.destroyAllWindows()


def _show_imshow_demo(stream_queue, server_queue, headless=False, show_latency=False, show_fps=False, visual=True,
                      overlay=False, debug=False):
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
    stream_viewer_thread_set = set()
    try:
        print("[INFO] imshow demo start")
        while True:
            stream = stream_queue.get()
            stream.sequence_perf_counter["demo_viewer_start"] = time.perf_counter()
            stream_name = stream.stream_name
            if stream_name not in stream_viewer_queue_dict:
                if debug: print(f"[DEBUG] {stream_name} is new in stream_viewer_queue_dict.")
                stream_viewer_queue_dict[stream_name] = Queue(maxsize=10)
                from threading import Thread
                viewer_thread = Thread(name=f"_update_imshow_process-{stream_name}", target=_update_imshow_process,
                                       args=(stream_viewer_queue_dict[stream_name], server_queue, headless,
                                             show_latency,
                                             show_fps, visual, overlay, debug))
                viewer_thread.daemon = True
                viewer_thread.start()
                stream_viewer_thread_set.add(viewer_thread)
                time.sleep(0.001)

            if stream_viewer_queue_dict[stream_name].full():
                print(f"[Warning] {stream_name} _show_imshow_demo queue is FULL.")

            stream_viewer_queue_dict[stream_name].put(stream)
            time.sleep(0.001)


    except KeyboardInterrupt:
        print("\nDEMO VIEWER is END by KeyboardInterrupt")

    except Exception as e:
        print(f"\nDEMO VIEWER is KILL by {e}")


def start_imshow_demo(stream_queue, server_queue=None, headless=False, show_latency=False, show_fps=False, visual=True,
                      overlay=False, debug=False, ):
    """
    imshow 데모를 백그라운드 스레드로 시작

    Args:
        :param overlay:
        :param visual: frame 합성 여부, False시 CLI Only(스트리밍도 불가)
        :param show_fps: FPS 출력 여부
        :param stream_queue: StreamFrameInstance 객체가 들어오는 메인 큐
        :param server_queue: None시
        :param debug: 디버그 메시지 출력 여부
        :param show_latency: 지연 시간 표시 여부
        :param headless: GUI창 미사용 여부

    Returns:
        Thread: 실행된 데모 스레드 객체

    """
    headless = headless or is_headless_cv2()

    imshow_demo_proc = Process(name="_show_imshow_demo", target=_show_imshow_demo,
                               args=(stream_queue, server_queue, headless, show_latency, show_fps, visual, overlay,
                                     debug))
    imshow_demo_proc.daemon = True
    imshow_demo_proc.start()
    return imshow_demo_proc


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
        gui_lines = [line for line in info.splitlines() if
                     "GUI:" in line or any(g in line for g in ["GTK", "Qt", "Win32", "Cocoa", "Carbon"])]
        for line in gui_lines:
            if "YES" in line or "ON" in line:
                return False  # GUI 지원됨
        return True  # GUI 백엔드가 모두 NO인 경우
    except Exception:
        return True  # 정보 조회 실패 → 보수적으로 headless로 간주

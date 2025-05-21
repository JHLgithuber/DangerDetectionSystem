from datetime import datetime
from pose_detector import crop_objects, draw_world_landmarks_with_coordinates
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from multiprocessing.managers import SharedMemoryManager
from threading import Thread
import torch

import dataclass_for_StreamFrameInstance
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

def visual_from_pose_estimation(stream_frame_instance, cls_conf=0.35):
    # 1. 원본 프레임 불러오기
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
        stream_frame_instance, debug=True)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))

    # 2. 객체별 crop 정보 구하기
    crop_object_images = crop_objects(stream_frame_instance, need_frame=False)

    # 3. 각 객체(사람)별로 스켈레톤 그린 overlay 오버레이 방식으로 합성
    for crop_object_img, pose_detection in zip(
            crop_object_images, stream_frame_instance.pose_detection_list):

        # (1) crop 크기만큼 검정 배경 생성
        #crop_h, crop_w = crop_object_img["crop"].shape[:2]
        #overlay = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        # (2) 오버레이에 스켈레톤(랜드마크) 그리기
        pose_landmark_overlay = draw_world_landmarks_with_coordinates(
            pose_detection, img_size=crop_object_img["img_size"],)

        # (3) bbox 좌표
        x1_p, y1_p, x2_p, y2_p = crop_object_img["bbox"]

        # (4) 원본 프레임의 해당 ROI 영역
        roi = frame[y1_p:y2_p, x1_p:x2_p]
        # (5) 마스크: overlay에서 검정색이 아닌 부분만 True
        mask = np.any(pose_landmark_overlay != [0, 0, 0], axis=2)
        # (6) ROI에 오버레이: mask 부분만 복사
        roi[mask] = pose_landmark_overlay[mask]
        # (7) (frame은 numpy view라서 자동 적용)

    return frame



def visual_from_detection_numpy(stream_frame_instance, cls_conf=0.35):
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

    vis_res = vis(row_img, bboxes, scores, cls, cls_conf, COCO_CLASSES)  # 프레임에 결과 그려줌
    return vis_res


def visual_from_tracking_serial(stream_frame_instance, cls_conf=0.35):
    # 프레임 복원
    frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance, debug=True)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))
    output = frame.copy()
    serialized_tracks = stream_frame_instance.human_tracking_serial

    for obj in serialized_tracks:
        score = obj["confidence"]
        if score < cls_conf:
            continue

        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class"]
        tid = obj["track_id"]

        label = f"{cls_name}#{tid} ({score:.2f})"
        color = (0, 255, 0)  # 필요시 cls_name에 따라 색 다르게

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output


def demo_viewer(stream_name, frame, debug=False):
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[ERROR] {stream_name}: 유효하지 않은 프레임")

        if frame.size == 0:
            print(f"[ERROR] {stream_name}: 빈 프레임")

        if debug: print(f"[DEBUG] {stream_name}: 프레임 shape: {frame.shape}, dtype: {frame.dtype}")
        cv2.imshow(stream_name, frame)

    except Exception as e:
        print(f"[ERROR] {stream_name} 뷰어 예외 발생: {e}")


def _add_latency_to_frame(frame, captured_datetime):
    """Calculate latency and add it to the frame."""
    delta = datetime.now() - captured_datetime
    latency_s = int(delta.total_seconds())
    latency_us = int(delta.total_seconds() * 1_000_000)
    latency_text = f"Latency is {latency_us:08d} us, about {latency_s}seconds"

    cv2.putText(
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
    cv2.putText(
        frame, 
        latency_text, 
        (20, 20),  # Top-left corner of the frame
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7,  # Font scale for better visibility
        (0, 0, 0),  # Black color
        1,  # Thickness
        cv2.LINE_AA
    )

def _update_imshow_process(stream_queue_for_process, show_latency=False, debug=False):
    stream_name = stream_queue_for_process.get().stream_name
    print(f"[INFO] {stream_name} imshow demo process start")
    try:
        cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(stream_name, 800, 600)

        sorter_gen=dataclass_for_StreamFrameInstance.sorter(messy_frame_instance_queue=stream_queue_for_process,
                                                            debug=debug)

        while True:
            #instances_per_frame_instance = stream_queue_for_process.get()
            instances_per_frame_instance=next(sorter_gen)
            if debug: 
                print(f"[DEBUG] {stream_name} instances_per_frame_instance is {instances_per_frame_instance}")
                
            if instances_per_frame_instance is not None:
                if instances_per_frame_instance.pose_detection_list is not None:
                    result_frame = visual_from_pose_estimation(
                        stream_frame_instance=instances_per_frame_instance,
                        cls_conf=0.35
                    )
                elif instances_per_frame_instance.human_tracking_serial is not None:
                    result_frame = visual_from_tracking_serial(
                        stream_frame_instance=instances_per_frame_instance, 
                        cls_conf=0.35
                    )
                elif instances_per_frame_instance.human_detection_numpy is not None:
                    result_frame = visual_from_detection_numpy(
                        stream_frame_instance=instances_per_frame_instance, 
                        cls_conf=0.35
                    )
                else:
                    result_frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
                        instances_per_frame_instance, 
                        debug=debug
                    )

                # Add latency to the frame
                if show_latency: _add_latency_to_frame(result_frame, instances_per_frame_instance.captured_datetime)

                # Display the updated frame
                demo_viewer(stream_name, result_frame, debug=debug)
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


def _show_imshow_demo(stream_queue, show_latency=False, debug=False):
    stream_viewer_queue_dict = dict()
    stream_viewer_process_set = set()
    try:
        # sorted_instance=dataclass_for_StreamFrameInstance.sorter(messy_frame_instance_queue=stream_queue, buffer_size=100)
        print("[INFO] imshow demo start")
        while True:
            # stream = next(sorted_instance)
            stream = stream_queue.get()
            stream_name = stream.stream_name
            if stream_name not in stream_viewer_queue_dict:
                if debug: print(f"[DEBUG] {stream_name} is new in stream_viewer_queue_dict.")
                stream_viewer_queue_dict[stream_name] = Queue()
                process = Process(target=_update_imshow_process, args=(stream_viewer_queue_dict[stream_name], show_latency, debug))
                # process.daemon = True
                stream_viewer_process_set.add(process)
                process.start()
                time.sleep(0.001)
            stream_viewer_queue_dict[stream_name].put(stream)
            #time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nDEMO VIEWER is END by KeyboardInterrupt")

    except Exception as e:
        print(f"\nDEMO VIEWER is KILL by {e}")

    finally:
        for viewer in stream_viewer_process_set:
            viewer.terminate()
            viewer.join()
            print(f"[INFO] {viewer.name} is terminated.")


def start_imshow_demo(stream_queue, show_latency=False, debug=False, ):
    imshow_demo_thread = Thread(target=_show_imshow_demo, args=(stream_queue, show_latency, debug))
    imshow_demo_thread.daemon = True
    imshow_demo_thread.start()
    return imshow_demo_thread
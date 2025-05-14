import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from threading import Thread
import torch

import dataclass_for_StreamFrameInstance
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

def visual_from_detection_numpy(stream_frame_instance, cls_conf=0.35):
    frame =dataclass_for_StreamFrameInstance.load_frame_to_shared_memory(stream_frame_instance.frame_info)
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
    frame =dataclass_for_StreamFrameInstance.load_frame_to_shared_memory(stream_frame_instance.frame_info)
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



def _update_imshow_process(stream_queue_for_process):
    stream_name = stream_queue_for_process.get().stream_name
    print(f"[INFO] {stream_name} imshow demo process start")
    try:
        cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(stream_name, 800, 600)
        while True:
            instances_per_frame_instance = stream_queue_for_process.get()
            if instances_per_frame_instance is not None:
                if instances_per_frame_instance.human_tracking_serial is not None:
                    print(f"[INFO] {stream_name} instances_per_frame is not None")
                    result_frame = visual_from_tracking_serial(stream_frame_instance=instances_per_frame_instance, cls_conf=0.35)

                elif instances_per_frame_instance.human_detection_numpy is not None:
                    result_frame = visual_from_detection_numpy(stream_frame_instance=instances_per_frame_instance, cls_conf=0.35)

                else:
                    result_frame = dataclass_for_StreamFrameInstance.load_frame_to_shared_memory(instances_per_frame_instance.frame_info)
                    result_frame = result_frame.reshape(
                        (instances_per_frame_instance.height, instances_per_frame_instance.width, 3))

                cv2.imshow(stream_name, result_frame)
            else:
                print(f"[INFO] {stream_name} instances_per_frame is None")

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return
    except Exception as e:
        cv2.destroyAllWindows()
        print(f"\nDEMO VIEWER of {stream_name} is KILL by {e}")
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print(f"DEMO VIEWER of {stream_name} is END by KeyboardInterrupt")
        return


stream_viewer_queue_dict = dict()
stream_viewer_process_set = set()


def _show_imshow_demo(stream_queue):
    try:
        sorted_instance=dataclass_for_StreamFrameInstance.sorter(messy_frame_instance_queue=stream_queue, buffer_size=100)
        while True:
            stream = next(sorted_instance)
            stream_name = stream.stream_name
            if stream_name not in stream_viewer_queue_dict:
                stream_viewer_queue_dict[stream_name] = Queue()
                process = Process(target=_update_imshow_process, args=(stream_viewer_queue_dict[stream_name],))
                process.daemon = True
                stream_viewer_process_set.add(process)
                process.start()
                time.sleep(0.001)
            stream_viewer_queue_dict[stream_name].put(stream)
            time.sleep(0.001)

    except KeyboardInterrupt:
        for viewer in stream_viewer_process_set:
            viewer.terminate()
            viewer.join()
        return
    except Exception as e:
        print(f"\nDEMO VIEWER is KILL by {e}")


def start_imshow_demo(stream_queue):
    imshow_demo_thread = Thread(target=_show_imshow_demo, args=(stream_queue,))
    imshow_demo_thread.daemon = True
    imshow_demo_thread.start()

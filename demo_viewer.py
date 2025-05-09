import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from threading import Thread
import torch
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

def visual(stream_frame_instance, cls_conf=0.35):
    frame = np.frombuffer(stream_frame_instance.row_frame_bytes, dtype=np.uint8)
    frame = frame.reshape((stream_frame_instance.height, stream_frame_instance.width, 3))
    test_size=(stream_frame_instance.human_detection_tsize, stream_frame_instance.human_detection_tsize)
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

    vis_res = vis(row_img, bboxes, scores, cls, cls_conf, COCO_CLASSES)   #프레임에 결과 그려줌
    return vis_res

def _update_imshow_process(stream_queue_for_process):
    stream_name = stream_queue_for_process.get().stream_name
    print(f"[INFO] {stream_name} imshow demo process start")
    try:
        while True:
            instances_per_frame_instance = stream_queue_for_process.get()
            if instances_per_frame_instance is not None:
                if instances_per_frame_instance.human_detection_numpy is not None:
                    result_frame = visual(stream_frame_instance=instances_per_frame_instance, cls_conf=0.35)
                    #print(f"[INFO] {stream_name} imshow demo process visual")
                else:
                    result_frame = np.frombuffer(instances_per_frame_instance.row_frame_bytes, dtype=np.uint8)
                    result_frame = result_frame.reshape((instances_per_frame_instance.height, instances_per_frame_instance.width, 3))
                    #print(f"[INFO] {stream_name} imshow demo process no visual")

                cv2.imshow(instances_per_frame_instance.stream_name, result_frame)
                cv2.waitKey(1)
            else:
                print(f"[INFO] {stream_name} instances_per_frame is None")

            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        while True:
            stream = stream_queue.get()
            stream_name = stream.stream_name
            if stream_name not in stream_viewer_queue_dict:
                stream_viewer_queue_dict[stream_name] = Queue()
                process = Process(target=_update_imshow_process, args=(stream_viewer_queue_dict[stream_name],))
                process.daemon = True
                stream_viewer_process_set.add(process)
                process.start()
            stream_viewer_queue_dict[stream_name].put(stream)

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


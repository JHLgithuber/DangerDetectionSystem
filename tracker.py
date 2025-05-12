import cv2
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import human_detector
import dataclass_for_StreamFrameInstance
import demo_viewer
from dataclass_for_StreamFrameInstance import StreamFrameInstance
import stream_input
from multiprocessing import Process, Manager
import queue
from collections import defaultdict
from yolox.data.datasets import COCO_CLASSES

def run_tracker_worker(messy_input_queue, output_queue, debug=False):
    tracker_proc = Process(target=_tracker_worker, args=(messy_input_queue, output_queue, debug))
    tracker_proc.daemon=True
    tracker_proc.start()
    return tracker_proc

def _tracker_worker(messy_input_queue, output_queue, debug):
    tracker=Tracker(messy_input_queue=messy_input_queue, messy_output_queue=output_queue, debug=debug)
    sorter=dataclass_for_StreamFrameInstance.sorter(messy_frame_instance_queue=messy_input_queue,
                                                    sorted_frame_instance_queue=None,
                                                    buffer_size=100)
    while True:
        input_inst=next(sorter)
        tracks=tracker.stream_updater(input_inst)
        human_tracking_serial=tracker.serialize_tracks(tracks)
        input_inst.human_tracking_serial=human_tracking_serial
        output_queue.put(input_inst)
        if debug:
            print("tracker_worker")
            print(
                f"input_inst.stream_name: {input_inst.stream_name}, input_inst.captured_datetime: {input_inst.captured_datetime}"
            )
            time.sleep(1/30)



class Tracker:
    trackers = defaultdict(lambda: DeepSort(max_age=30, n_init=3))
    stream_dict=dict()

    def __init__(self, messy_input_queue, messy_output_queue, debug=False):
        self.messy_input_queue=messy_input_queue
        self.messy_output_queue=messy_output_queue
        self.debug=debug

    def stream_updater(self, frame_inst):
        h, w = frame_inst.height, frame_inst.width
        one_d = np.frombuffer(frame_inst.row_frame_bytes, dtype=np.uint8)
        frame = one_d.reshape((h, w, 3))
        detections = self._to_deepsort_dets(frame_inst.human_detection_numpy)
        tracks = (self.trackers[frame_inst.stream_name] #스트림별로 다른 트래커 사용
                  .update_tracks(detections, frame=frame))
        if self.debug:
            print("stream_updater")
        return tracks

    def _to_deepsort_dets(self, np_arr):
        """
        np_arr: N×7 배열 [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        """
        if np_arr is None or len(np_arr) == 0:
            return []

        # 1) bbox, score, class_id 분리
        boxes     = np_arr[:, :4]                                     # N×4
        scores    = (np_arr[:, 4] * np_arr[:, 5]).astype(float)       # N
        class_ids = np_arr[:, 6].astype(int)                          # N

        # 2) DeepSORT 포맷으로 변환
        detections = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
            w, h      = x2 - x1, y2 - y1
            cls_name  = COCO_CLASSES[cid]  # YOLOX가 알려준 class_id → 문자열
            detections.append(([x1, y1, w, h], score, cls_name))

        if self.debug:
            print("Save deepsort_dets")

        return detections

    def serialize_tracks(self, tracks):
        """
        Track 객체 리스트 → 순수 Python 타입 리스트(dict)로 변환
        """

        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            if self.debug: print(f"Serialized Track: Track ID={t.track_id}, BBox=({x1}, {y1}, {x2}, {y2})")
            out.append({
                "track_id": t.track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": getattr(t, 'score', 0.0),
                "class": getattr(t, 'det_class', 'undefined')
            })
        if out==[]:
            out=None
        if self.debug: print(f"Final Serialized Output: {out}")
        return out


if __name__=="__main__":
    args = human_detector.get_args()
    exp = human_detector.get_exp(args.exp_file, args.name)

    debugMode = True
    # showMode=True
    stream_queue = Manager().Queue(maxsize=32)
    return_from_detect_queue = Manager().Queue(maxsize=32)
    return_queue = Manager().Queue(maxsize=32)


    demo_viewer.start_imshow_demo(stream_queue=return_queue)
    time.sleep(1)

    detector_process = human_detector.main(exp, args, stream_queue, return_from_detect_queue,)
    detector_process.start()
    time.sleep(3)

    tracker_worker= run_tracker_worker(return_from_detect_queue, return_queue, debug=debugMode)

    testStreamList = [
        stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=stream_queue,
                                stream_name="TEST_0", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_1", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_2", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_3", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_4", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_5", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_6", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_7", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_8", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_9", debug=debugMode),
    ]
    detector_process.join()
    tracker_worker.join()
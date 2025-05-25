import time
import numpy as np
from multiprocessing import Process
from HybridSORT.trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from dataclass_for_StreamFrameInstance import sorter
from types import SimpleNamespace


def run_tracker_worker(messy_input_queue, output_queue, debug=False):
    p = Process(target=_tracker_worker, name="tracker_worker", args=(messy_input_queue, output_queue, debug))
    p.daemon = True
    p.start()
    if debug: print("[Tracker] started")
    return p


def _tracker_worker(messy_input_queue, output_queue, debug):
    # 스트림별 HybridSort 트래커 저장소
    args = SimpleNamespace(
        track_thresh=0.5,  # 칼만 필터 추적시 최소 신뢰도
        TCM_first_step=False,  # 첫 번째 association 단계 TCM 사용 여부
        TCM_byte_step=False,  # BYTE association 사용 여부
        TCM_byte_step_weight=0.1  # BYTE association 가중치
    )
    trackers = {}
    sorted_frame_gen = sorter(messy_input_queue, debug=debug)
    while True:
        frame_inst = next(sorted_frame_gen)  # dataclass_for_StreamFrameInstance 인스턴스
        #frame_inst = messy_input_queue.get()
        if debug: print(f"[Tracker] received frame_inst: {frame_inst}")
        name = frame_inst.stream_name

        # ① 스트림별로 트래커 초기화
        if name not in trackers:
            trackers[name] = Hybrid_Sort(
                args=args,
                det_thresh=0.5,
                max_age=30,
                iou_threshold=0.7,
                #with_reid=False
            )
            if debug: print(f"[Tracker:{name}] init")
        tracker = trackers[name]

        # ② detection numpy → [x1,y1,x2,y2,score] 배열로 변환
        dets_np = frame_inst.human_detection_numpy
        if dets_np is None or len(dets_np) == 0:
            output_queue.put(frame_inst)
            continue
        elif dets_np.shape[1] >= 6:
            boxes = dets_np[:, :4]  # (x1,y1,x2,y2)
            obj_conf = dets_np[:, 4]
            class_conf = dets_np[:, 5]
            scores = obj_conf * class_conf

            dets = np.hstack([boxes, scores[:, None]])
        else:
            raise ValueError(f"Unexpected dets_np shape: {dets_np.shape}")


        # ③ tracker.update → tracks: (M,6) [x1,y1,x2,y2,track_id,score]
        img_size = (frame_inst.height, frame_inst.width)
        try:
            tracks = tracker.update(dets, img_size, img_size)
        except Exception as e:
            print(f"[Tracker:{name}] update error: {e}")
            frame_inst.human_tracking_serial = None
            output_queue.put(frame_inst)
            continue

        # ④ 결과 직렬화
        serialized = []
        for x1, y1, x2, y2, track_id, score in tracks:
            serialized.append({
                "track_id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(score)
            })

        frame_inst.human_tracking_serial = serialized or None
        output_queue.put(frame_inst)

        if debug:
            print(f"[Tracker:{name}] tracks={serialized}")
        time.sleep(1 / 60)

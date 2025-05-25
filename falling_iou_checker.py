# pipeline.py

import numpy as np
from collections import deque
from multiprocessing import Process, Queue
from dataclass_for_StreamFrameInstance import StreamFrameInstance
from fall_detecting_algorithm import detect_fall
from pose_detector import crop_objects
#from pose_worker import run_pose_worker
#from output_worker import run_output_worker

def compute_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - interA
    return interA/union if union>0 else 0.0


def _fall_worker(in_q: Queue, out_q: Queue,
                 buffer_size, iou_thresh, fall_ratio_thresh, debug):
    # stream_name → deque[list of {'bbox','is_flag'}]
    histories = {}

    while True:
        frame: StreamFrameInstance = in_q.get()
        name = frame.stream_name
        history = histories.setdefault(name, deque(maxlen=buffer_size))

        crops = crop_objects(frame, need_frame=False)
        current_data = []
        fall_flags = []

        if frame.pose_detection_list is None:
            print(f"[Fall_IoU] {name}pose_detection_list is None")
            out_q.put(frame)

        for i, crop in enumerate(crops):
            bbox = tuple(crop['bbox'])
            pose_det = frame.pose_detection_list[i]
            is_flag = detect_fall(pose_det)

            # 과거 모든 프레임의 모든 bbox와 비교
            match_cnt = fall_cnt = 0
            for past_frame in history:
                for past in past_frame:
                    if compute_iou(past['bbox'], bbox) >= iou_thresh:
                        match_cnt += 1
                        if past['is_flag']:
                            fall_cnt += 1


            # 매칭된 프레임 중 fall 비율로 최종 판단
            if match_cnt > 0 and (fall_cnt/match_cnt) >= fall_ratio_thresh:
                fall_flags.append(True)
                if debug: print(f"[Fall_IoU] {name} idx={i} FALL")
            else:
                fall_flags.append(False)
                if debug: print(f"[Fall_IoU] {name} idx={i} Not FALL")

            current_data.append({'bbox': bbox, 'is_flag': is_flag})
            if debug: print(f"[Fall_IoU] current_data=> {name} idx={i} is_flag={is_flag}")

        # 버퍼에 추가하고 결과 저장
        history.append(current_data)
        frame.fall_flag_list = fall_flags
        out_q.put(frame)

def run_fall_worker(input_q: Queue, output_q: Queue,
                    buffer_size=30, iou_thresh=0.5, fall_ratio_thresh=0.6,
                    debug=False) -> Process:
    p = Process(
        target=_fall_worker,
        name="fall_worker",
        args=(input_q, output_q, buffer_size, iou_thresh, fall_ratio_thresh, debug)
    )
    p.daemon = True
    p.start()
    if debug: print("[Fall] worker started")
    return p

if __name__ == "__main__":
    q_input          = Queue(maxsize=100)
    q_pose_to_fall   = Queue(maxsize=100)
    q_fall_to_output = Queue(maxsize=100)

    pose_p = run_pose_worker(q_input,        q_pose_to_fall,   debug=True)
    fall_p = run_fall_worker(q_pose_to_fall, q_fall_to_output, debug=True)
    out_p  = run_output_worker(q_fall_to_output, debug=True)

    # frame 생성 후 q_input.put(frame) 만 하면
    # pose → fall → output 순으로 처리됩니다.

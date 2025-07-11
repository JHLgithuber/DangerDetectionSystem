from collections import deque
from multiprocessing import Process, Queue

from dataclass_for_StreamFrameInstance import StreamFrameInstance
from fall_detecting_algorithm import detect_fall
from pose_detector import crop_objects


# noinspection PyPep8Naming
def compute_iou(a, b):
    """
    두 바운딩박스 간 IoU(Intersection over Union) 계산

    Args:
        a (tuple or list): 첫 번째 박스 (x1, y1, x2, y2)
        b (tuple or list): 두 번째 박스 (x1, y1, x2, y2)

    Returns:
        float: IoU 값 (0.0 ~ 1.0)
    """
    xA = max(a[0], b[0]);
    yA = max(a[1], b[1])
    xB = min(a[2], b[2]);
    yB = min(a[3], b[3])
    interW = max(0, xB - xA);
    interH = max(0, yB - yA)
    interA = interW * interH
    areaA = (a[2] - a[0]) * (a[3] - a[1]);
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - interA
    return interA / union if union > 0 else 0.0


def _fall_worker(in_q: Queue, out_q: Queue, buffer_size, iou_thresh, fall_ratio_thresh, debug=False):
    """
    낙상 여부를 IoU 기반으로 누적 판단하는 워커 프로세스

    Args:
        in_q (Queue): StreamFrameInstance 입력 큐
        out_q (Queue): 낙상 여부 판단 후 출력 큐
        buffer_size (int): 프레임당 낙상 기록 유지 길이
        iou_thresh (float): 바운딩박스 IoU 매칭 기준
        fall_ratio_thresh (float): 과거 낙상 비율 기준 (낙상 개수 / 매칭 개수)
        debug (bool): 디버그 출력 여부

    Returns:
        None
    """
    histories = {}
    while True:
        frame: StreamFrameInstance = in_q.get()
        name = frame.stream_name
        # 스트림별 히스토리 생성 및 삽입, 오래된 항목 자동 삭제
        history = histories.setdefault(name, deque(maxlen=buffer_size))

        crops = crop_objects(frame, need_frame=False)
        current_data = []
        fall_flags = []

        # 포즈 감지 결과가 있는지 여부
        if frame.pose_detection_list is None:
            if debug: print(f"[Fall_IoU] {name}pose_detection_list is None")
            frame.fall_flag_list = None
            out_q.put(frame)
            continue

        for i, crop in enumerate(crops):
            bbox = tuple(crop['bbox'])
            pose_det = frame.pose_detection_list[i]
            is_flag = detect_fall(pose_det, debug=debug)
            if debug: print(f"[Fall_IoU] {name} idx={i} is_flag={is_flag}")

            # 과거 frame의 모든 bbox와 비교
            match_cnt = fall_cnt = 0
            for past_frame in history:
                for past in past_frame:
                    if compute_iou(past['bbox'], bbox) >= iou_thresh:
                        match_cnt += 1
                        if past['is_flag']:
                            fall_cnt += 1

            # 매칭된 프레임 중 fall 비율로 최종 판단
            if match_cnt > 0 and (fall_cnt / match_cnt) >= fall_ratio_thresh:
                fall_flags.append(True)
                if debug: print(f"[Fall_IoU] {name} idx={i} FALL")
            else:
                fall_flags.append(False)
                if debug: print(f"[Fall_IoU] {name} idx={i} Not FALL")

            current_data.append({'bbox': bbox, 'is_flag': is_flag})
            if debug: print(f"[Fall_IoU] current_data=> {name} idx={i} is_flag={is_flag}")

        # 버퍼에 추가 하고 결과 저장
        history.append(current_data)
        frame.fall_flag_list = fall_flags
        out_q.put(frame)


def run_fall_worker(input_q: Queue, output_q: Queue,
                    buffer_size=30, iou_thresh=0.5, fall_ratio_thresh=0.6,
                    debug=False):
    """
    낙상 판정 워커 프로세스를 백그라운드로 실행

    Args:
        input_q (Queue): StreamFrameInstance 입력 큐
        output_q (Queue): 낙상 결과 전달 큐
        buffer_size (int): 각 스트림당 프레임 기록 길이
        iou_thresh (float): IoU 기반 bbox 매칭 임계값
        fall_ratio_thresh (float): 누적된 낙상 비율 판단 기준
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        Process: 시작된 워커 프로세스 객체
    """
    p = Process(
        target=_fall_worker,
        name="fall_worker",
        args=(input_q, output_q, buffer_size, iou_thresh, fall_ratio_thresh, debug)
    )
    p.daemon = True
    p.start()
    if debug: print("[Fall] worker started")
    return p

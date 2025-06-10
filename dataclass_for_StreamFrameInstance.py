import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class StreamFrameInstance:
    stream_name: str
    frame_index: int
    memory_name:str
    height: int
    width: int
    captured_datetime: datetime = field(default_factory=datetime.now)
    human_detection_numpy: Optional[np.ndarray] = None
    # noinspection SpellCheckingInspection
    human_detection_tsize: int = 640
    human_tracking_serial: Optional[List[Dict[str, Any]]] = None
    pose_detection_list: Optional[np.ndarray] = None
    fall_flag_list: Optional[List[bool]] = None
    bypass_flag: bool = False


def save_frame_to_shared_memory(frame, shm_name, debug=False):
    shm = None
    try:
        shm = SharedMemory(name=shm_name)
        buffer=np.ndarray(frame.shape, dtype=np.uint8, buffer=shm.buf)
        np.copyto(buffer, frame)
        shm.close()
        if debug: print(f"save {shm.name} to shared memory")
        return shm_name
    except Exception as e:
        print(f"공유 메모리 저장 중 오류: {e}")
        return None
    finally:
        shm.close()


def load_frame_from_shared_memory(stream_frame_instance, debug=False):
    """공유 메모리에서 프레임을 로드합니다."""
    if debug: 
        print(f"[DEBUG] load_frame_from_shared_memory start")
    memory_name = stream_frame_instance.memory_name
    shm = None
    try:
        if debug: print(f"memory_name is {memory_name} for load_frame")
        shm = shared_memory.SharedMemory(name=memory_name)
        shape = (stream_frame_instance.height, stream_frame_instance.width, 3)
        # 즉시 복사하여 새로운 버퍼 생성
        frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
        shm.close()
        if debug: 
            print(f"load {stream_frame_instance.stream_name} frame from shared memory")
        return frame

    except Exception as e:
        print(f"공유 메모리 로드 중 오류: {e}")
        black_frame = np.zeros((stream_frame_instance.height, stream_frame_instance.width, 3), dtype=np.uint8)
        return black_frame
        
    finally:
        # 항상 공유 메모리 연결을 닫습니다
        if shm is not None:
            shm.close()


def sorter(messy_frame_instance_queue, sorted_frame_instance_queue=None, buffer_size=30, debug=False,):
    # noinspection SpellCheckingInspection
    """
        시간순으로 정렬된 프레임 인스턴스를 생성하는 제너레이터

        Args:
            messy_frame_instance_queue: 정렬되지 않은 입력 큐
            sorted_frame_instance_queue: 정렬된 결과를 넣을 출력 큐 (선택 사항)
            buffer_size: 정렬에 사용할 버퍼 크기

        Yields:
            시간순으로 정렬된 StreamFrameInstance
            :param sorted_frame_instance_queue:
            :param messy_frame_instance_queue:
            :param debug: 디버그
        """
    stream_buffers = defaultdict(list)
    if debug: print(f"[DEBUG] sorter is run")
    while True:
        # 큐에서 데이터 가져오기
        # noinspection SpellCheckingInspection
        try:
            instance = messy_frame_instance_queue.get(timeout=0.5)
            
            # 스트림별로 분리하여 버퍼링
            stream_buffers[instance.stream_name].append(instance)
            
            # 각 스트림별 버퍼가 일정 크기를 넘으면 가장 오래된 프레임 제공
            for stream_name, instances in list(stream_buffers.items()):
                if len(instances) > buffer_size:  # 각 스트림에 버퍼 크기의 10% 할당
                    # 날짜 기준 정렬
                    instances.sort(key=lambda x: x.captured_datetime)
                    oldest = instances.pop(0)  # 가장 오래된 항목 제거

                    if debug: print(f"[DEBUG] sorted_frame_instance_queue.put(oldest_image)")
                    # 제너레이터로 반환
                    if sorted_frame_instance_queue:
                        sorted_frame_instance_queue.put(oldest)
                    yield oldest

        except Exception as e:
            if debug: print(f"[DEBUG] sorter error or empty: {e}")
            # 큐가 비어있거나 다른 예외 발생 시 잠시 대기
            time.sleep(0.01)
            
            # 버퍼에 데이터가 있으면 가장 오래된 프레임 제공
            all_empty = True
            for stream_name, instances in list(stream_buffers.items()):
                if instances:
                    all_empty = False
                    instances.sort(key=lambda x: x.captured_datetime)
                    oldest = instances.pop(0)
                    
                    if sorted_frame_instance_queue:
                        sorted_frame_instance_queue.put(oldest)
                    yield oldest
            
            # 모든 버퍼가 비었을 때 더 오래 대기
            if all_empty:
                time.sleep(0.1)
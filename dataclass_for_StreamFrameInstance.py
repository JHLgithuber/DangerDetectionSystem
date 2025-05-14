from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import time
import numpy as np
from multiprocessing import shared_memory
from queue import Queue
import heapq
from collections import defaultdict

@dataclass
class StreamFrameInstance:
    stream_name: str
    frame_info: Dict[str, Any]  # 공유 메모리 정보 (name, shape, dtype)
    height: int
    width: int
    captured_datetime: datetime = field(default_factory=datetime.now)
    human_detection_numpy: Optional[np.ndarray] = None
    human_detection_tsize: int = 640
    human_tracking_serial: Optional[List[Dict[str, Any]]] = None
    bypass_flag: bool = False
    
    def __del__(self):
        """객체가 파괴될 때 자동으로 호출되어 공유 메모리를 정리합니다."""
        try:
            # 공유 메모리 해제 시도
            if hasattr(self, 'frame_info') and self.frame_info and 'name' in self.frame_info:
                try:
                    shm = shared_memory.SharedMemory(name=self.frame_info['name'], create=False)
                    shm.close()
                    shm.unlink()  # 참조 카운트를 적절히 관리하기 위해 조심해서 사용
                except Exception as e:
                    # 이미 해제되었거나 다른 프로세스가 사용 중인 경우 무시
                    pass
        except Exception:
            # 종료 과정에서 발생하는 예외는 무시
            pass


def load_frame_to_shared_memory(frame_info):
    """공유 메모리에서 프레임을 로드합니다."""
    try:
        if not frame_info or 'name' not in frame_info:
            return None
        
        shm = shared_memory.SharedMemory(name=frame_info['name'], create=False)
        shape = frame_info['shape']
        dtype = np.dtype(frame_info['dtype'])
        
        # 공유 메모리에서 배열 생성 (복사 없이)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        
        # 데이터 복사본 만들기 (공유 메모리 의존성 제거)
        result = np.copy(array)
        
        # 사용 후 공유 메모리 연결 해제 (삭제하지 않음)
        shm.close()
        
        return result
    except Exception as e:
        print(f"공유 메모리 로드 중 오류: {e}")
        return None


def sorter(messy_frame_instance_queue, sorted_frame_instance_queue=None, buffer_size=100):
    """
    시간순으로 정렬된 프레임 인스턴스를 생성하는 제너레이터
    
    Args:
        messy_frame_instance_queue: 정렬되지 않은 입력 큐
        sorted_frame_instance_queue: 정렬된 결과를 넣을 출력 큐 (선택 사항)
        buffer_size: 정렬에 사용할 버퍼 크기
    
    Yields:
        시간순으로 정렬된 StreamFrameInstance
    """
    buffer = []
    stream_buffers = defaultdict(list)
    
    while True:
        # 큐에서 데이터 가져오기
        try:
            instance = messy_frame_instance_queue.get(timeout=0.1)
            
            # 스트림별로 분리하여 버퍼링
            stream_buffers[instance.stream_name].append(instance)
            
            # 각 스트림별 버퍼가 일정 크기를 넘으면 가장 오래된 프레임 제공
            for stream_name, instances in list(stream_buffers.items()):
                if len(instances) > buffer_size // 10:  # 각 스트림에 버퍼 크기의 10% 할당
                    # 날짜 기준 정렬
                    instances.sort(key=lambda x: x.captured_datetime)
                    oldest = instances.pop(0)  # 가장 오래된 항목 제거
                    
                    # 제너레이터로 반환
                    if sorted_frame_instance_queue:
                        sorted_frame_instance_queue.put(oldest)
                    yield oldest
        
        except Exception as e:
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
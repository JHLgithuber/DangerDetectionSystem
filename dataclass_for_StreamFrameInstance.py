import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class StreamFrameInstance:
    """
    스트리밍 프레임 단위의 메타데이터 및 분석 결과를 저장하는 데이터 클래스.

    Attributes:
        stream_name (str): 해당 프레임이 소속된 스트림의 이름이자 식별자.
        frame_index (int): 해당 스트림 내에서의 프레임 순번.
        memory_name (str): 공유 메모리에서 프레임을 참조하기 위한 메모리 이름.
        height (int): 프레임의 세로 해상도.
        width (int): 프레임의 가로 해상도.
        captured_time (int): 프레임이 수집된 시각. 기본값은 생성 시점의 현재 시간. time.time_ns()
        human_detection_numpy (Optional[np.ndarray]): YOLOX 등에서 추출한 사람 탐지 결과(numpy 배열).
        human_detection_tsize (int): 탐지 시 사용된 입력 이미지의 크기 (YOLOX의 --tsize).
        human_tracking_serial (Optional[List[Dict[str, Any]]]): 트래킹 알고리즘 결과로 얻은 사람 식별 정보.
        pose_detection_list (Optional[np.ndarray]): MediaPipe 등에서 추출한 포즈 인식 결과 리스트.
        fall_flag_list (Optional[List[bool]]): 포즈 기반 낙상 여부 판단 결과 리스트.
        bypass_flag (bool): 해당 프레임이 분석 대상에서 제외되어 우회되었는지 여부.
    """
    stream_name: str
    frame_index: int
    memory_name: str
    height: int
    width: int
    captured_time: int = field(default_factory=time.time_ns, )
    human_detection_numpy: Optional[np.ndarray] = None
    # noinspection SpellCheckingInspection
    human_detection_tsize: int = 640
    human_tracking_serial: Optional[List[Dict[str, Any]]] = None
    pose_detection_numpy: Optional[np.ndarray] = None
    fall_flag_list: Optional[List[bool]] = None
    bypass_flag: bool = False
    sequence_perf_counter: dict = None


def save_frame_to_shared_memory(frame, shm_name, debug=False):
    """
    주어진 영상 프레임을 지정된 이름의 공유 메모리에 저장

    Args:
        frame (np.ndarray): 저장할 영상 프레임 (dtype=np.uint8).
        shm_name (str): 공유 메모리 블록 이름.
        debug (bool): True일 경우 저장 성공 메시지를 출력

    Returns:
        str or None: 성공적으로 저장되면 공유 메모리 이름을 반환하며,
                     오류 발생 시 None을 반환

    Raises:
        Exception: 공유 메모리 접근 또는 복사 중 예외처리.
    """
    shm = None
    try:
        shm = SharedMemory(name=shm_name)
        buffer = np.ndarray(frame.shape, dtype=np.uint8, buffer=shm.buf)
        np.copyto(buffer, frame)
        shm.close()
        if debug: print(f"save {shm.name} to shared memory")
        return shm_name
    except Exception as e:
        print(f"공유 메모리 저장 중 오류: {e}")
        return None
    finally:
        shm.close()


def load_frame_from_shared_memory(stream_frame_instance, copy=True, debug=False):
    """
    공유 메모리에서 프레임 로드

    Args:
        :param stream_frame_instance: 프레임 메타정보 포함 객체
        :param copy: 복사 여부(프레임 복사시 성능 하락, 편집 필요시 복사)
        :param debug: : 디버그 메시지 출력 여부

    Returns:
        np.ndarray: 로드된 프레임, 실패 시 검정 프레임 반환
    """

    if debug:
        print(f"[DEBUG] load_frame_from_shared_memory start")
    memory_name = stream_frame_instance.memory_name
    shm = None
    try:
        if debug: print(f"memory_name is {memory_name} for load_frame")
        shm = shared_memory.SharedMemory(name=memory_name)
        shape = (stream_frame_instance.height, stream_frame_instance.width, 3)
        # 즉시 복사하여 새로운 버퍼 생성
        if copy:
            if debug: print(f"copy {stream_frame_instance.stream_name} frame from shared memory")
            frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
            shm.close()
            return frame
        else:
            if debug: print(f"load {stream_frame_instance.stream_name} frame from shared memory")
            frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
            return frame, shm

    except Exception as e:
        print(f"공유 메모리 로드 중 오류: {e}")
        black_frame = np.zeros((stream_frame_instance.height, stream_frame_instance.width, 3), dtype=np.uint8)
        return black_frame

    finally:
        # 항상 공유 메모리 연결을 닫음
        if copy and shm is not None:
            shm.close()


class FrameSequentialProcesser:
    def __init__(self, processing_data_name:str, perf_counter_name="FrameSequential" ,debug=False):
        self.__debug = debug
        self.__metadata_dict=dict()
        self.__processing_data_name=processing_data_name
        #self.__lock = threading.Lock()
        #self.__cond = threading.Condition(self.__lock)
        self.__perf_counter_name = perf_counter_name+"_"
        if self.__debug: print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} init")


    def metadata_push(self,input_metadata:StreamFrameInstance):
        input_metadata.sequence_perf_counter[self.__perf_counter_name+"start"]=time.perf_counter()
        data_id = input_metadata.captured_time + hash(input_metadata.stream_name)
        #with self.__cond:
        self.__metadata_dict[data_id] = {"metadata" : input_metadata, "push_perf_counter" : time.perf_counter(), "processing" : False}
        if self.__debug: print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} metadata_push {data_id}: {input_metadata}")
        #    self.__cond.notify_all()
        return data_id

    def metadata_pop(self, max_buf_time=30 , loop_wait_time=0.0001, blocking=False) -> StreamFrameInstance or None:
        #with self.__cond:
        while True:
            # 1) 버퍼가 비어 있으면 대기
            if not self.__metadata_dict:
                if blocking:
                    #self.__cond.wait(loop_wait_time)
                    time.sleep(loop_wait_time)
                    continue
                else:
                    return None

            # 2) FIFO 순서의 가장 오래된 항목 조회
            key, meta = next(iter(self.__metadata_dict.items()))

            if time.perf_counter() - meta["push_perf_counter"] > max_buf_time:
                self.__metadata_dict.pop(key)
                if self.__debug:
                    print(f"[Warning] FrameSequentialProcesser {self.__processing_data_name} metadata_pop ignore {key}")
                #self.__cond.notify_all()
                continue

            # 3) 값이 채워졌으면 꺼내서 반환
            if meta["processing"]:
                output = self.__metadata_dict.pop(key)["metadata"]
                output.sequence_perf_counter[self.__perf_counter_name+"end"]=time.perf_counter()
                if self.__debug:
                    print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} metadata_pop {key}: {output}")
                #self.__cond.notify_all()
                return output

            if blocking:
                pass
                # 4) 값이 아직 없으면 대기
                #self.__cond.wait(loop_wait_time)
            else:
                return None

    def processing_value_input(self, data_id:int, value) -> None:
        #with self.__cond:
        try:
            setattr(self.__metadata_dict[data_id]["metadata"], self.__processing_data_name, value)
            self.__metadata_dict[data_id]["processing"] = True
            if self.__debug:
                print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} processing_value_input {data_id} ← {value}")
        except KeyError:
            print(f"[ERROR] FrameSequentialProcesser {self.__processing_data_name} processing_value_input KEY ERROR")
        finally:
            pass
            # 값이 채워졌으니 pop 대기 깨우기
            #self.__cond.notify_all()

    def data_id_in_buffer(self,data_id:int) -> bool:
        #with self.__cond:
        if self.__debug: print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} data_id_in_buffer {data_id} in buffer: {data_id in self.__metadata_dict}")
        #self.__cond.notify_all()
        return data_id in self.__metadata_dict

    def is_buffer_empty(self) -> bool:
        #with self.__cond:
        if not self.__metadata_dict:
            status= True
        else:
            status= False
        if self.__debug: print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} is_buffer_empty: {status}")
        #self.__cond.notify_all()
        return status

    def is_oldest_finsh(self) -> bool:
        #with self.__cond:
        if self.is_buffer_empty():
            #self.__cond.notify_all()
            return False
        else:
            key, meta = next(iter(self.__metadata_dict.items()))
            if self.__debug: print(f"[DEBUG] FrameSequentialProcesser {self.__processing_data_name} is_oldest_finsh oldest_finsh: {meta['processing']}")
            #self.__cond.notify_all()
            return meta["processing"]


def sorter(messy_frame_instance_queue, sorted_frame_instance_queue=None, buffer_size=10, debug=False, ):
    # noinspection SpellCheckingInspection
    """
        시간순으로 정렬된 프레임 인스턴스를 생성하는 제너레이터

        Args:
            messy_frame_instance_queue: 정렬되지 않은 입력 큐
            sorted_frame_instance_queue: 정렬된 결과를 넣을 출력 큐 (선택 사항)
            buffer_size: 정렬에 사용할 버퍼 크기

        Yields:
            시간순으로 정렬된 StreamFrameInstance
            :param buffer_size: 스트림당 버퍼 사이즈, 커질수록 레이턴시가 증가
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
            instance.sequence_perf_counter["instance_sorter_start"] = time.perf_counter()

            # 스트림별로 분리하여 버퍼링
            stream_buffers[instance.stream_name].append(instance)

            # 각 스트림별 버퍼가 일정 크기를 넘으면 가장 오래된 프레임 제공
            for stream_name, instances in list(stream_buffers.items()):
                if len(instances) > buffer_size:
                    # 날짜 기준 정렬
                    instances.sort(key=lambda x: x.captured_time)
                    oldest = instances.pop(0)  # 가장 오래된 항목 제거

                    if debug: print(f"[DEBUG] sorted_frame_instance_queue.put(oldest_image)")
                    oldest.sequence_perf_counter["instance_sorter_end"] = time.perf_counter()
                    # 제너레이터로 반환
                    if sorted_frame_instance_queue:
                        sorted_frame_instance_queue.put(oldest)
                    yield oldest

        except Exception as e:
            if debug: print(f"[DEBUG] sorter error or empty: {e}")

            # 버퍼에 데이터가 있으면 가장 오래된 프레임 제공
            for stream_name, instances in list(stream_buffers.items()):
                if instances:
                    instances.sort(key=lambda x: x.captured_time)
                    oldest = instances.pop(0)

                    if sorted_frame_instance_queue:
                        sorted_frame_instance_queue.put(oldest)
                    yield oldest


def compute_time_deltas(timing_log: dict) -> dict:
    """
    입력 딕셔너리의 key 순서대로 시간 차(ms)를 계산하여 반환합니다.

    Args:
        timing_log (dict): 시간 로그 딕셔너리 (key 순서 보장)

    Returns:
        dict: {"이전_key → 현재_key": 경과시간(ms)} 형식의 딕셔너리
    """
    keys = list(timing_log.keys())
    deltas = {}
    for i in range(1, len(keys)):
        prev_key = keys[i - 1]
        curr_key = keys[i]
        delta_ms = (timing_log[curr_key] - timing_log[prev_key]) * 1000
        deltas[f"{prev_key}\t→ {curr_key}"] = delta_ms
    return deltas

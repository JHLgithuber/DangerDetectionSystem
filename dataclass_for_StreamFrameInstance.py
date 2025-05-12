from dataclasses import dataclass, field
from datetime import datetime
import numpy


# 각각의 프레임의 정보 이것 저것 넣을 데이터클래스
@dataclass
class StreamFrameInstance:
    stream_name: str
    captured_datetime: datetime = field(init=False)
    row_frame_bytes: bytes
    height: int
    width: int
    bypass_flag: bool = False

    human_detection_numpy: numpy = None
    human_detection_tsize: int = 640

    human_tracking_serial: list = None

    # 해당 프레임의 정보 이것 저것 라벨링 이라던가?

    def __post_init__(self):
        self.captured_datetime = datetime.now()


def sorter(messy_frame_instance_queue, sorted_frame_instance_queue=None, buffer_size=50):
    buffer = []
    while True:
        if len(buffer) < buffer_size:
            buffer.append(messy_frame_instance_queue.get())
        else:
            buffer.sort(key=lambda x: x.captured_datetime)
            if sorted_frame_instance_queue is None:
                yield buffer.pop(0)
            else:
                sorted_frame_instance_queue.put(buffer.pop(0))
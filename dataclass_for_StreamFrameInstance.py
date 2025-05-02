from dataclasses import dataclass, field
from datetime import datetime


#각각의 프레임의 정보 이것 저것 넣을 데이터클래스
@dataclass
class StreamFrameInstance:
    stream_name: str
    captured_datetime: datetime = field(init=False)
    row_frame_bytes: bytes
    #해당 프레임의 정보 이것 저것 라벨링 이라던가?

    def __post_init__(self):
        self.captured_datetime=datetime.now()
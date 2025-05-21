import argparse
import sys
import time
import uuid
from multiprocessing import Manager

import cv2

import dataclass_for_StreamFrameInstance
from threading import Thread

import av
import numpy as np

import demo_viewer
from dataclass_for_StreamFrameInstance import StreamFrameInstance


def _process_frames_common(
        frame_iterator,
        stream_name,
        shm_names,
        metadata_queue,
        debug,
        bypass_frame,
        receive_frame,
        ignore_frame,
        stream_start_index,
):
    bypassed_count = 0
    received_count = receive_frame
    ignore_count = 0
    index = stream_start_index

    for frame in frame_iterator:
        raw_stream_view = np.array(frame.to_ndarray(format='bgr24'))

        if ignore_count > 0:
            ignore_count -= 1
            if ignore_count == 0:
                received_count = receive_frame
            if debug: print(f"[{stream_name}] 무시")
            continue

        if debug: print(f"[{stream_name}] 수신: {raw_stream_view.shape}, 평균 밝기: {raw_stream_view.mean():.2f}")

        if bypassed_count < bypass_frame:
            bypassed_count += 1
            bypass_flag = True
        else:
            bypass_flag = False
            bypassed_count = 0

        memory_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
            frame=raw_stream_view,
            shm_name=shm_names[index],
            debug=debug
        )
        if memory_name is None:
            continue

        stream_frame_instance = StreamFrameInstance(
            stream_name=stream_name,
            frame_index=index,
            memory_name=memory_name,
            height=raw_stream_view.shape[0],
            width=raw_stream_view.shape[1],
            bypass_flag=bypass_flag,
        )

        index = (index + 1) % len(shm_names)
        received_count -= 1
        if received_count <= 0:
            ignore_count = ignore_frame

        if metadata_queue.full():
            metadata_queue.get()
        metadata_queue.put(stream_frame_instance)
        time.sleep(1 / 30)


def _update_frame_from_rtsp(rtsp_url, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame, ignore_frame, stream_start_index, ):
    try:
        print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
        container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
        frame_iterator = container.decode(video=0)
        _process_frames_common(
            frame_iterator, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame, ignore_frame, stream_start_index,
        )
    except Exception as e:
        print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")


def _update_frame_from_file(rtsp_url, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame, ignore_frame, stream_start_index):
    try:
        while True:
            print(f"[INFO] Video File: {rtsp_url} will OPEN")
            container = av.open(rtsp_url)
            frame_iterator = container.decode(video=0)
            _process_frames_common(
                frame_iterator, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame, ignore_frame, stream_start_index
            )
            print("endVideo")
            container.close()
    except Exception as e:
        print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")


class RtspStream:
    def __init__(self, rtsp_url, metadata_queue, stream_name=str(uuid.uuid4()), bypass_frame=0, receive_frame=1,
                 ignore_frame=0, debug=False, is_file=False):
        self.manager_smm = None
        self.stream_start_index=None
        self.stream_thread = None
        self.rtsp_url = rtsp_url
        self.is_file = is_file
        self.stream_name = stream_name
        self.metadata_queue = metadata_queue
        self.debug = debug
        self.bypass_frame = bypass_frame
        self.receive_frame = receive_frame
        self.ignore_frame = ignore_frame
        self.shape = None
        self.bytes = None

        self._get_first_frame()

    def _get_first_frame(self):
        try:
            container = av.open(self.rtsp_url, options={'rtsp_transport': 'tcp'})
            frame = next(container.decode(video=0))
            img = frame.to_ndarray(format='bgr24')
            self.shape = img.shape
            self.bytes = img.nbytes
            print(f"[INFO] RTSP URL: {self.rtsp_url} will OPEN for initialization")

        except Exception as e:
            print(f"[ERROR] {self.stream_name} 초기화 프레임 확보 실패: {e}")

    def get_stream_name(self):
        return self.stream_name

    def get_shape(self):
        return self.shape

    def get_bytes(self):
        return self.bytes

    def _stream_slow_starting_up(self):
        start_percent = 0
        index = 0
        max_frame_count = 20
        empty_frame = None
        for index in range(max_frame_count):
            start_percent=index/max_frame_count*100
            empty_frame = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
            y0 = 100     # 첫 줄의 y좌표
            dy = 80      # 줄 간 간격 (원하는 만큼 조절)
            lines = [
                "Hello!",
                "Processing stream is starting up...",
                "Please wait about a minute.",
                f"{start_percent:.2f}%"
            ]
            for i, line in enumerate(lines):
                y = y0 + i * dy
                cv2.putText(
                    empty_frame,
                    line,
                    (50, y),  # x, y좌표
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

            memory_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
                frame=empty_frame.copy(),
                shm_name=self.manager_smm[index],
                debug=self.debug
            )

            stream_frame_instance = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=index,
                memory_name=memory_name,
                height=self.shape[0],
                width=self.shape[1],
                bypass_flag=False,
            )

            index = (index + 1) % len(self.manager_smm)
            print(f"stream start up... {index}")
            self.metadata_queue.put(stream_frame_instance)
            time.sleep(max((max_frame_count-index)/4,0.5))
        return index


    def run_stream(self, manager_smm):
        self.manager_smm=manager_smm
        self.stream_start_index= self._stream_slow_starting_up()
        if not self.is_file:
            self.stream_thread = Thread(target=_update_frame_from_rtsp, name=self.stream_name,
                                        args=(self.rtsp_url, self.stream_name, self.manager_smm, self.metadata_queue, self.debug,
                                              self.bypass_frame, self.receive_frame, self.ignore_frame, self.stream_start_index,))
        else:
            self.stream_thread = Thread(target=_update_frame_from_file, name=self.stream_name,
                                        args=(self.rtsp_url, self.stream_name, self.manager_smm, self.metadata_queue, self.debug,
                                                  self.bypass_frame, self.receive_frame, self.ignore_frame, self.stream_start_index,))
        self.stream_thread.daemon = True
        self.stream_thread.start()
        return self.stream_thread

    def kill_stream(self):
        self.__del__()

    def __del__(self):
        if self.stream_thread.is_alive():
            self.metadata_queue.put(None)
            # TODO: 파이프로 스트림종료 명시
            # self.stream_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="디폴트 값 테스트")
    parser.add_argument("--debug", type=bool, default=False, help="디버그 로그 출력")
    parser.add_argument("--show", type=bool, default=True, help="스트림 화면 출력")
    args = parser.parse_args()

    debugMode = args.debug
    showMode = args.show
    queue = Manager().Queue(maxsize=512)

    testStreamList = [
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=queue, stream_name="TEST_0",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", manager_queue=queue, stream_name="TEST_1",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", manager_queue=queue, stream_name="TEST_2",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", manager_queue=queue, stream_name="TEST_3",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", manager_queue=queue, stream_name="TEST_4",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", manager_queue=queue, stream_name="TEST_5",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", manager_queue=queue, stream_name="TEST_6",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", manager_queue=queue, stream_name="TEST_7",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", manager_queue=queue, stream_name="TEST_8",
                   debug=debugMode),
        RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", manager_queue=queue, stream_name="TEST_9",
                   debug=debugMode), ]

    try:
        if showMode:
            demo_viewer.start_imshow_demo(stream_queue=queue)
        for testStream in testStreamList:
            testStream.stream_thread.join()
    except KeyboardInterrupt:
        for testStream in testStreamList:
            del testStream
            for viewer_process in demo_viewer.stream_viewer_process_set:
                viewer_process.terminate()
                viewer_process.join()
        print("\nEND by KeyboardInterrupt")
        sys.exit(0)

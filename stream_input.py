import time
import uuid
from threading import Thread
import cv2
import numpy as np

import dataclass_for_StreamFrameInstance
from dataclass_for_StreamFrameInstance import StreamFrameInstance


class RtspStream:
    def __init__(self, rtsp_url, metadata_queue, stream_name=str(uuid.uuid4()), bypass_frame=0, receive_frame=1,
                 ignore_frame=0, startup_max_frame_count=60, debug=False, media_format="rtsp", resize=None,
                 startup_pass=False):
        self.startup_max_frame_count = startup_max_frame_count
        self.startup_pass = startup_pass
        self.running = True
        self.manager_smm = None
        self.stream_start_index = None
        self.stream_thread = None
        self.rtsp_url = rtsp_url
        self.format = media_format
        self.stream_name = stream_name
        self.metadata_queue = metadata_queue
        self.resize = resize
        self.debug = debug
        self.bypass_frame = bypass_frame
        self.receive_frame = receive_frame
        self.ignore_frame = ignore_frame
        self.shape = None
        self.bytes = None

        self._get_first_frame()

    def _get_first_frame(self):
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("첫 프레임을 읽을 수 없습니다.")
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)
            self.shape = frame.shape
            self.bytes = frame.nbytes
            print(f"[INFO] Source: {self.rtsp_url} will OPEN for initialization")
        except Exception as e:
            print(f"[ERROR] {self.stream_name} 초기화 프레임 확보 실패: {e}")

    def get_stream_name(self):
        return self.stream_name

    def get_shape(self):
        return self.shape

    def get_bytes(self):
        return self.bytes

    def _stream_slow_starting_up(self):
        index = 0
        for index in range(self.startup_max_frame_count):
            if not self.running:
                break
            sequence_perf_counter = {"stream_input_start": time.perf_counter()}
            start_percent = index / self.startup_max_frame_count * 100
            empty_frame = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
            if self.resize is not None:
                empty_frame = cv2.resize(empty_frame, self.resize)

            lines = [
                "Hello!",
                "Processing stream is starting up...",
                "Please wait about a minute.",
                f"{start_percent:.2f}%"
            ]
            for i, line in enumerate(lines):
                cv2.putText(empty_frame, line, (50, 100 + i * 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            memory_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
                frame=empty_frame.copy(), shm_name=self.manager_smm[index], debug=self.debug)

            sequence_perf_counter["stream_input_end"] = time.perf_counter()
            stream_frame_instance = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=index,
                memory_name=memory_name,
                height=empty_frame.shape[0],
                width=empty_frame.shape[1],
                bypass_flag=False,
                sequence_perf_counter=sequence_perf_counter)

            index = (index + 1) % len(self.manager_smm)
            self.metadata_queue.put(stream_frame_instance)
            time.sleep(0.1 if self.startup_pass else 3)
        return index

    def run_stream(self, manager_smm):
        self.manager_smm = manager_smm
        self.stream_thread = Thread(target=self._capture_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        return self.stream_thread

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        index = self._stream_slow_starting_up()
        bypassed_count = 0
        received_count = self.receive_frame
        ignore_count = 0
        frame_last_input_time = 0

        while self.running and cap.isOpened():
            sequence_perf_counter = {"stream_input_start": time.perf_counter()}
            ret, frame = cap.read()
            if not ret:
                continue

            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            if ignore_count > 0:
                ignore_count -= 1
                if ignore_count == 0:
                    received_count = self.receive_frame
                continue

            if bypassed_count < self.bypass_frame:
                bypassed_count += 1
                bypass_flag = True
            else:
                bypass_flag = False
                bypassed_count = 0

            memory_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
                frame=frame, shm_name=self.manager_smm[index], debug=self.debug)
            if memory_name is None:
                continue

            sequence_perf_counter["stream_input_end"] = time.perf_counter()
            stream_frame_instance = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=index,
                memory_name=memory_name,
                height=frame.shape[0],
                width=frame.shape[1],
                bypass_flag=bypass_flag,
                sequence_perf_counter=sequence_perf_counter)

            index = (index + 1) % len(self.manager_smm)
            received_count -= 1
            if received_count <= 0:
                ignore_count = self.ignore_frame

            if self.metadata_queue.full():
                self.metadata_queue.get()
            self.metadata_queue.put(stream_frame_instance)
            frame_last_input_time = time.perf_counter()

        cap.release()

    def kill_stream(self):
        self.running = False
        return self.stream_thread

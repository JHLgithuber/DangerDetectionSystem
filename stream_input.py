import time
import uuid
from threading import Thread
import cv2
import numpy as np

import dataclass_for_StreamFrameInstance
from dataclass_for_StreamFrameInstance import StreamFrameInstance


class RtspStream:
    """
    RTSP, 파일, 웹캠 등 영상을 프레임 단위로 가져오는 스트림 클래스
    PyAV를 제거하고 OpenCV VideoCapture로 구현
    """

    def __init__(
        self,
        rtsp_url,
        metadata_queue,
        stream_name=None,
        bypass_frame=0,
        receive_frame=1,
        ignore_frame=0,
        startup_max_frame_count=60,
        debug=False,
        media_format="rtsp",
        file_fps=None,
        resize=None,
        startup_pass=False,
    ):
        # 외부 파라미터와 내부 필드 매핑
        self.rtsp_url = rtsp_url
        self.metadata_queue = metadata_queue
        self.stream_name = stream_name or str(uuid.uuid4())
        self.bypass_frame = bypass_frame
        self.receive_frame = receive_frame
        self.ignore_frame = ignore_frame
        # 기존 startup_max_frame_count 파라미터 활용
        self.startup_dummy_count = startup_max_frame_count
        # startup_pass 플래그가 True면 딜레이 스킵
        self.skip_startup_delay = startup_pass
        self.debug = debug
        self.media_format = media_format.lower()
        self.resize = resize

        if self.media_format == "file" and file_fps is not None:
            self.input_fps = file_fps
        elif self.media_format == "file":
            self.input_fps = 30
        else:
            self.input_fps = None

        self.running = True
        self.manager_smm = None
        self.stream_thread = None

        # 첫 프레임 확보 및 해상도, 바이트 계산
        self._get_initial_frame()

    def _get_initial_frame(self):
        try:
            cap = self.__cv2_capture()
            ret, img = cap.read()
            cap.release()
            if not ret or img is None:
                raise RuntimeError("첫 프레임 읽기 실패")
            self.height, self.width = img.shape[:2]
            self.frame_bytes = img.nbytes
            if self.debug:
                print(f"[INFO] Initialized frame: {self.height}x{self.width}, bytes={self.frame_bytes}")
        except Exception as e:
            print(f"[ERROR] {self.stream_name} 초기화 실패: {e}")

    def get_stream_name(self):
        return self.stream_name

    def get_shape(self):
        return self.height, self.width

    def get_bytes(self):
        return self.frame_bytes

    def _frame_generator(self, cap):
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame
        cap.release()

    def _enqueue(self, instance):
        if self.metadata_queue.full():
            try:
                self.metadata_queue.get_nowait()
            except:
                pass
        self.metadata_queue.put(instance)

    def _send_startup_dummy(self):
        for i in range(self.startup_dummy_count):
            if not self.running:
                return i
            seq = {"stream_input_start": time.perf_counter()}
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if self.resize and (self.resize[0] < self.width or self.resize[1] < self.height):
                frame = cv2.resize(frame, self.resize)
            cv2.putText(
                frame,
                f"Loading {i+1}/{self.startup_dummy_count}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )
            shm = self.manager_smm[i]
            mem_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
                frame=frame.copy(), shm_name=shm, debug=self.debug
            )
            seq["stream_input_end"] = time.perf_counter()
            inst = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=i,
                memory_name=mem_name,
                height=self.height,
                width=self.width,
                bypass_flag=False,
                sequence_perf_counter=seq,
            )
            self._enqueue(inst)
            if not self.skip_startup_delay:
                time.sleep(1)
        return self.startup_dummy_count - 1

    def run_stream(self, manager_smm):
        self.manager_smm = manager_smm
        if self.media_format == "file":
            target = self._loop_stream
        else:
            target = self._once_stream
        self.stream_thread = Thread(target=target, name=self.stream_name, daemon=True)
        self.stream_thread.start()
        return self.stream_thread

    def _once_stream(self):
        start_idx = self._send_startup_dummy()
        cap = self.__cv2_capture()
        gen = self._frame_generator(cap)
        self._process_loop(gen, start_idx)

    def _loop_stream(self):
        start_idx = self._send_startup_dummy()
        while self.running:
            cap = self.__cv2_capture()
            gen = self._frame_generator(cap)
            self._process_loop(gen, start_idx)
            if self.running and self.debug:
                print("[INFO] Reopening stream...")


    def __cv2_capture(self):
        if self.media_format == "rtsp":
            cap = cv2.VideoCapture(self.rtsp_url)
        elif self.media_format == "file":
            cap = cv2.VideoCapture(self.rtsp_url)
        elif self.media_format == "webcam_id":
            cap = cv2.VideoCapture(int(self.rtsp_url))
        else:
            cap = cv2.VideoCapture(self.rtsp_url)
        return cap

    def _process_loop(self, frame_iter, start_idx):
        idx = start_idx
        bypassed = 0
        received = self.receive_frame
        ignored = 0
        last_time = 0
        for frame in frame_iter:
            gap = 1 / self.input_fps if self.input_fps and self.media_format == 'file' else 0
            while gap > time.perf_counter() - last_time:
                pass
            seq = {"stream_input_start": time.perf_counter()}
            if not self.running:
                break
            img = frame
            if ignored > 0:
                ignored -= 1
                if ignored == 0:
                    received = self.receive_frame
                continue
            if bypassed < self.bypass_frame:
                bypassed += 1
                bypass = True
            else:
                bypass = False
                bypassed = 0
            if self.resize and (self.resize[0] < self.width or self.resize[1] < self.height):
                img = cv2.resize(img, self.resize)
            mem = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(
                frame=img, shm_name=self.manager_smm[idx], debug=self.debug
            )
            seq["stream_input_end"] = time.perf_counter()
            inst = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=idx,
                memory_name=mem,
                height=img.shape[0],
                width=img.shape[1],
                bypass_flag=bypass,
                sequence_perf_counter=seq,
            )
            self._enqueue(inst)
            idx = (idx + 1) % len(self.manager_smm)
            received -= 1
            if received <= 0:
                ignored = self.ignore_frame
            last_time = time.perf_counter()

    def kill_stream(self):
        self.running = False
        return self.stream_thread

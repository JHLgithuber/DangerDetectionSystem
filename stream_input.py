import time
import uuid
from threading import Thread
import av
import cv2
import numpy as np

import dataclass_for_StreamFrameInstance
from dataclass_for_StreamFrameInstance import StreamFrameInstance


class RtspStream:
    """
    RTSP, 파일, 웹캠 등 영상을 프레임 단위로 가져오는 스트림 클래스

    Args:
        rtsp_url (str): 영상 소스 (RTSP URL 또는 파일 경로)
        metadata_queue (Queue): 프레임 메타데이터 전달용 큐
        stream_name (str): 스트림 고유 이름
        bypass_frame (int): 건너뛸 프레임 수
        receive_frame (int): 연속 수신할 프레임 수
        ignore_frame (int): 연속 무시할 프레임 수
        startup_max_frame_count (int): 초기 지연 시 출력할 더미 프레임 수
        debug (bool): 디버그 출력 여부
        media_format (str): 소스 형식 ('rtsp', 'file', 커스텀 포맷 등)
    """

    # TODO: Resize 추가

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
        """
        영상 스트림에서 첫 프레임을 읽어 해상도 및 메모리 크기 초기화

        동작:
            - RTSP/파일/기타 포맷 구분하여 AV 컨테이너 열기
            - 첫 프레임 디코딩 후 shape, bytes 저장
        """
        try:
            if self.format == "rtsp":
                container = av.open(self.rtsp_url, options={'rtsp_transport': 'tcp'})
            elif self.format == "file":
                container = av.open(self.rtsp_url)
            else:  # 카메라 등
                container = av.open(self.rtsp_url, format=self.format)
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
        """
        스트림 시작 전 더미 프레임을 생성하여 전체 초기화

        Returns:
            int: 마지막으로 사용된 공유 메모리 인덱스

        동작:
            - 일정 시간 동안 텍스트가 포함된 검정 프레임 생성
            - 공유 메모리에 저장 후 메타데이터 큐에 전송
            - 진행률(%) 표시 및 sleep 간격으로 로딩 연출
        """
        index = 0
        for index in range(self.startup_max_frame_count):
            if not self.running:
                break
            sequence_perf_counter = {"stream_input_start": time.perf_counter()}
            start_percent = index / self.startup_max_frame_count * 100
            empty_frame = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)

            if (
                    self.resize is not None and
                    (self.resize[0] < self.shape[1] or self.resize[1] < self.shape[0])  # 가로나 세로 중 하나라도 더 작으면
            ):
                empty_frame = cv2.resize(empty_frame, self.resize)

            y0 = 100  # 첫 줄의 y좌표
            dy = 80  # 줄 간 간격 (원하는 만큼 조절)
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

            sequence_perf_counter["stream_input_end"] = time.perf_counter()
            stream_frame_instance = StreamFrameInstance(
                stream_name=self.stream_name,
                frame_index=index,
                memory_name=memory_name,
                height=self.shape[0],
                width=self.shape[1],
                bypass_flag=False,
                sequence_perf_counter=sequence_perf_counter,
            )

            index = (index + 1) % len(self.manager_smm)
            print(f"stream start up... {index}")
            self.metadata_queue.put(stream_frame_instance)
            if not self.startup_pass:
                time.sleep(3)
            else:
                time.sleep(0.1)
        return index

    def startup_pass(self):
        self.startup_pass = True
        print(f"{self.stream_name}startup PASS")

    def run_stream(self, manager_smm, ):
        """
        스트림 유형에 따라 프레임 수신 스레드 실행

        Args:
            manager_smm (list): 공유 메모리 이름 리스트

        Returns:
            Thread: 실행된 스트림 처리 스레드 객체

        동작:
            - RTSP, 파일, 커스텀 포맷에 따라 적절한 처리 함수 선택
            - 스레드를 데몬 모드로 시작
            - 스트림명으로 스레드 이름 지정
        """
        self.manager_smm = manager_smm
        if self.format == "rtsp":
            self.stream_thread = Thread(target=self._update_frame_from_rtsp, name=self.stream_name,
                                        args=(self.rtsp_url, self.stream_name, self.manager_smm, self.metadata_queue,
                                              self.debug,
                                              self.bypass_frame, self.receive_frame, self.ignore_frame,
                                              ))
        elif self.format == "file":
            self.stream_thread = Thread(target=self._update_frame_from_file, name=self.stream_name,
                                        args=(self.rtsp_url, self.stream_name, self.manager_smm, self.metadata_queue,
                                              self.debug,
                                              self.bypass_frame, self.receive_frame, self.ignore_frame,
                                              ))
        else:
            self.stream_thread = Thread(target=self._update_frame_from_custom_format, name=self.stream_name,
                                        args=(self.rtsp_url, self.stream_name, self.manager_smm, self.metadata_queue,
                                              self.debug,
                                              self.bypass_frame, self.receive_frame, self.ignore_frame,
                                              )
                                        )
        self.stream_thread.daemon = True
        self.stream_thread.start()
        return self.stream_thread

    def _process_frames_common(
            self,
            frame_iterator,
            stream_name,
            shm_names,
            metadata_queue,
            debug,
            bypass_frame,
            receive_frame,
            ignore_frame,
            start_index,
            input_fps=None,
    ):
        """
        공통 프레임 처리 루프 (RTSP, 파일, 카메라 등 공용)

        Args:
            frame_iterator: AV 프레임 반복자
            stream_name (str): 스트림 이름
            shm_names (list): 공유 메모리 이름 리스트
            metadata_queue (Queue): 메타데이터 전송 큐
            debug (bool): 디버그 메시지 출력 여부
            bypass_frame (int): 주기적으로 처리를 스킵할 프레임 수
            receive_frame (int): 받아들일 프레임 수
            ignore_frame (int): 이후 무시할 프레임 수
            start_index (int): 공유 메모리 시작 인덱스

        Returns:
            None

        동작:
            - 프레임을 bgr24 배열로 변환하여 공유 메모리에 저장
            - 주기적 bypass / ignore 프레임 제어
            - 메타데이터 큐에 StreamFrameInstance 전송
        """
        bypassed_count = 0
        received_count = receive_frame
        ignore_count = 0
        frame_time_gap = 0
        frame_last_input_time = 0
        index = start_index

        for frame in frame_iterator:
            if input_fps is not None:
                frame_time_gap= 1/input_fps
            else:
                frame_time_gap = 0.0

            while frame_time_gap > time.perf_counter() - frame_last_input_time: # 정밀한 FPS구현
                pass

            sequence_perf_counter = {"stream_input_start": time.perf_counter()}
            if not self.running:
                break
            raw_stream_view = np.array(frame.to_ndarray(format='bgr24'))

            if ignore_count > 0:  # 프레임 무시 처리
                ignore_count -= 1
                if ignore_count == 0:
                    received_count = receive_frame
                if debug: print(f"[{stream_name}] 무시")
                continue

            if debug: print(f"[{stream_name}] 수신: {raw_stream_view.shape}, 평균 밝기: {raw_stream_view.mean():.2f}")

            if bypassed_count < bypass_frame:  # 프레임 통과 처리
                bypassed_count += 1
                bypass_flag = True
            else:
                bypass_flag = False
                bypassed_count = 0

            if (
                    self.resize is not None and
                    (self.resize[0] < self.shape[1] or self.resize[1] < self.shape[0])  # 가로나 세로 중 하나라도 더 작으면
            ):
                raw_stream_view = cv2.resize(raw_stream_view, self.resize)

            memory_name = dataclass_for_StreamFrameInstance.save_frame_to_shared_memory(  # 공유메모리에 프레임 저장
                frame=raw_stream_view,
                shm_name=shm_names[index],
                debug=debug
            )
            if memory_name is None:
                continue

            sequence_perf_counter["stream_input_end"] = time.perf_counter()
            stream_frame_instance = StreamFrameInstance(  # 메타데이터 인스턴스 생성
                stream_name=stream_name,
                frame_index=index,
                memory_name=memory_name,
                height=raw_stream_view.shape[0],
                width=raw_stream_view.shape[1],
                bypass_flag=bypass_flag,
                sequence_perf_counter= sequence_perf_counter,
            )

            index = (index + 1) % len(shm_names)
            received_count -= 1
            if received_count <= 0:  # 프레임 수신 처리
                ignore_count = ignore_frame

            if metadata_queue.full():  # 큐 다차면 가장 과거 데이터 삭제
                metadata_queue.get()
            metadata_queue.put(stream_frame_instance)

            frame_last_input_time = time.perf_counter()
            time.sleep(0)


    def _update_frame_from_rtsp(self, rtsp_url, stream_name, shm_names, metadata_queue, debug, bypass_frame,
                                receive_frame, ignore_frame, ):
        """
        RTSP 스트림에서 프레임을 수신하고 처리 루프 실행

        Args:
            rtsp_url (str): RTSP 주소
            stream_name (str): 스트림 식별자
            shm_names (list): 공유 메모리 이름 리스트
            metadata_queue (Queue): 메타데이터 전송 큐
            debug (bool): 디버그 출력 여부
            bypass_frame (int): 건너뛸 프레임 수
            receive_frame (int): 수신할 프레임 수
            ignore_frame (int): 무시할 프레임 수

        Returns:
            None

        Raises:
            Exception: RTSP 연결, 디코딩 또는 처리 중 예외 발생 (내부 print로 출력)
        """
        try:
            start_index = self._stream_slow_starting_up()
            print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
            container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
            frame_iterator = container.decode(video=0)
            self._process_frames_common(
                frame_iterator, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame,
                ignore_frame, start_index,
            )
        except Exception as e:
            print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")

    def _update_frame_from_file(self, rtsp_url, stream_name, shm_names, metadata_queue, debug, bypass_frame,
                                receive_frame, ignore_frame, ):
        """
        비디오 파일에서 프레임을 반복적으로 읽어 처리

        Args:
            rtsp_url (str): 파일 경로
            stream_name (str): 스트림 식별자
            shm_names (list): 공유 메모리 이름 리스트
            metadata_queue (Queue): 메타데이터 전송 큐
            debug (bool): 디버그 출력 여부
            bypass_frame (int): 건너뛸 프레임 수
            receive_frame (int): 수신할 프레임 수
            ignore_frame (int): 무시할 프레임 수

        Returns:
            None

        Raises:
            Exception: 파일 열기, 디코딩, 처리 중 예외 발생 (내부에서 출력 처리)
        """
        try:
            start_index = self._stream_slow_starting_up()
            while True:
                if not self.running:
                    break
                print(f"[INFO] Video File: {rtsp_url} will OPEN")
                container = av.open(rtsp_url)
                frame_iterator = container.decode(video=0)
                self._process_frames_common(
                    frame_iterator, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame,
                    ignore_frame, start_index, input_fps= 30,
                )
                print("endVideo")
                container.close()

        except Exception as e:
            print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")

    def _update_frame_from_custom_format(self, rtsp_url, stream_name, shm_names, metadata_queue, debug, bypass_frame,
                                         receive_frame, ignore_frame, ):
        """
        카메라 등에서 프레임을 반복적으로 읽어 처리

        Args:
            rtsp_url (str): 장치 등 경로
            stream_name (str): 스트림 식별자
            shm_names (list): 공유 메모리 이름 리스트
            metadata_queue (Queue): 메타데이터 전송 큐
            debug (bool): 디버그 출력 여부
            bypass_frame (int): 건너뛸 프레임 수
            receive_frame (int): 수신할 프레임 수
            ignore_frame (int): 무시할 프레임 수

        Returns:
            None

        Raises:
            Exception: 파일 열기, 디코딩, 처리 중 예외 발생 (내부에서 출력 처리)
        """
        try:
            start_index = self._stream_slow_starting_up()
            while True:
                if not self.running:
                    break
                print(f"[INFO] Video File: {rtsp_url} will OPEN")
                container = av.open(rtsp_url, format=self.format)
                frame_iterator = container.decode(video=0)
                self._process_frames_common(
                    frame_iterator, stream_name, shm_names, metadata_queue, debug, bypass_frame, receive_frame,
                    ignore_frame, start_index, input_fps= 30,
                )
                print("endVideo")
                container.close()

        except Exception as e:
            print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")

    def kill_stream(self):
        self.running = False
        return self.stream_thread

from dataclass_for_StreamFrameInstance import StreamFrameInstance
import av
import cv2
import numpy as np
import threading
import uuid
from multiprocessing import Queue, Process

def _update_frame(rtsp_url, stream_name, stream_queue):
    print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
    container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
    try:
        for frame in container.decode(video=0):
            raw_stream_view = frame.to_ndarray(format='bgr24')
            print(f"[{stream_name}] 수신: {raw_stream_view.shape}, 평균 밝기: {raw_stream_view.mean():.2f}")
            stream_frame_instance = StreamFrameInstance(
                stream_name=stream_name, row_frame_bytes=raw_stream_view.tobytes())

            stream_queue.put(stream_frame_instance)

    except Exception as e:
        print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")
        container.close()

class RtspStream:
    def __init__(self, rtsp_url, stream_name=str(uuid.uuid4())):
        self.rawStreamView = None
        self.rtsp_url = rtsp_url

        #self.stream = self.container.streams.video[0]
        self.stream_name = stream_name

        # self.stream_thread = threading.Thread(target=self.__update_frame)
        # self.stream_thread.daemon = True
        # self.stream_thread.start()
        #self.stream_thread.join()

        self.stream_queue = Queue()
        self.stream_process = Process(target=self._update_frame, args=(self.rtsp_url, self.stream_name, self.stream_queue))
        self.stream_process.daemon=True
        self.stream_process.start()

    def get_stream_name(self):
        return self.stream_name
    def get_stream_queue(self):
        return self.stream_queue

    def __del__(self):
        self.stream_process.terminate()


instances_of_imshow_demo = []
def show_imshow_demo(stream_queue):
    def update_imshow_process(stream_queue_for_process):
        while True:
            instances_per_frame=stream_queue_for_process.get()
            if instances_per_frame is not None:
                img_bytes=instances_per_frame.row_frame_bytes
                frame=np.frombuffer(img_bytes, dtype=np.uint8)
                frame=frame.reshape((instances_per_frame.height, instances_per_frame.width, 3))
                cv2.imshow(instances_per_frame.stream_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


    imshow_demo_process = Process(target=update_imshow_process, args=stream_queue)
    imshow_demo_process.daemon = True
    imshow_demo_process.start()
    instances_of_imshow_demo.append(imshow_demo_process)


if __name__ == "__main__":
    testStream0=RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", stream_name="TEST_0")
    testStream1 = RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", stream_name="TEST_1")
    testStream2 = RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", stream_name="TEST_2")

    show_imshow_demo(stream_queue=testStream0.get_stream_queue())
    show_imshow_demo(stream_queue=testStream1.get_stream_queue())
    show_imshow_demo(stream_queue=testStream2.get_stream_queue())



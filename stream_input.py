import av
import cv2
import numpy as np
import threading
import uuid


class RtspStream:
    def __init__(self, rtsp_url, stream_name=str(uuid.uuid4())):
        self.rawStreamView = None
        self.rtsp_url = rtsp_url
        print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
        self.container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
        self.stream = self.container.streams.video[0]
        self.stream_name = stream_name

        self.stream_thread = threading.Thread(target=self.__update_frame)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        #self.stream_thread.join()

    def get_raw_stream_bytes(self):
        pass

    def view_live_stream(self):
        pass

    def __update_frame(self):
        try:
            for frame in self.container.decode(video=0):
                self.rawStreamView = frame.to_ndarray(format='bgr24')
                print(f"[{self.stream_name}] 수신: {self.rawStreamView.shape}, 평균 밝기: {self.rawStreamView.mean():.2f}")
        except Exception as e:
            print(f"[ERROR] {self.stream_name} 스레드 예외 발생: {e}")

    def __del__(self):
        cv2.destroyWindow(self.stream_name)
        self.container.close()

def show_all_streams_imshow_demo(instances):
    while True:
        for instance in instances:
            frame = instance.rawStreamView
            if frame is not None and frame.shape[0] > 0:
                cv2.imshow(instance.stream_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    testStream0=RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", stream_name="TEST_0")
    testStream1 = RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", stream_name="TEST_1")
    #testStream2 = RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", stream_name="TEST_2")
    show_all_streams_imshow_demo([testStream0,testStream1])


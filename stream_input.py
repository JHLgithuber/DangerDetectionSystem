import sys
import time
from dataclass_for_StreamFrameInstance import StreamFrameInstance
import av
import argparse
import cv2
import numpy as np
import uuid
from multiprocessing import Queue, Process


def _update_frame(rtsp_url, stream_name, stream_queue, debug=False):
    try:
        print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
        container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
        for frame in container.decode(video=0):
                raw_stream_view = frame.to_ndarray(format='bgr24')
                if debug: print(f"[{stream_name}] 수신: {raw_stream_view.shape}, 평균 밝기: {raw_stream_view.mean():.2f}")
                stream_frame_instance = StreamFrameInstance(
                    stream_name=stream_name,
                    row_frame_bytes=raw_stream_view.tobytes(),
                    height=raw_stream_view.shape[0],
                    width=raw_stream_view.shape[1]
                )

                if stream_queue.full():
                    stream_queue.get()
                stream_queue.put_nowait(stream_frame_instance)
                time.sleep(0.01)

    except Exception as e:
        print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")
        #container.close()


class RtspStream:
    def __init__(self, rtsp_url, stream_name=str(uuid.uuid4()), debug=False):
        self.rawStreamView = None
        self.rtsp_url = rtsp_url
        self.stream_name = stream_name
        self.debug=debug


        self.stream_queue = Queue(maxsize=3600)
        self.stream_process = Process(target=_update_frame,
                                      args=(self.rtsp_url, self.stream_name, self.stream_queue, self.debug))
        self.stream_process.daemon = True
        self.stream_process.start()

    def get_stream_name(self):
        return self.stream_name

    def get_stream_queue(self):
        return self.stream_queue

    def __del__(self):
        if self.stream_process.is_alive():
            self.stream_queue.put(None)
            self.stream_process.terminate()
            self.stream_process.join()

instances_of_imshow_demo = []
def _update_imshow_process(stream_queue_for_process):
    print(f"[INFO] imshow demo process start")
    try:
        while True:
            instances_per_frame = stream_queue_for_process.get()
            if instances_per_frame is not None:
                img_bytes = instances_per_frame.row_frame_bytes
                frame = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = frame.reshape((instances_per_frame.height, instances_per_frame.width, 3))
                cv2.imshow(instances_per_frame.stream_name, frame)
                time.sleep(0.01)
            else:
                print(f"[INFO] instances_per_frame is None")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print(f"\n{stream_queue_for_process.get().stream_name} is END by KeyboardInterrupt")
        cv2.destroyAllWindows()
        raise KeyboardInterrupt

def show_imshow_demo(stream_queue):
    imshow_demo_process = Process(target=_update_imshow_process, args=(stream_queue,))
    imshow_demo_process.daemon = True
    try:
        imshow_demo_process.start()
        instances_of_imshow_demo.append(imshow_demo_process)
    except KeyboardInterrupt:
        imshow_demo_process.terminate()
        imshow_demo_process.join()
        raise KeyboardInterrupt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="디폴트 값 테스트")
    parser.add_argument("--debug", type=bool, default=False, help="디버그 로그 출력")
    parser.add_argument("--show", type=bool, default=True, help="스트림 화면 출력")
    args = parser.parse_args()

    debugMode=args.debug
    showMode=args.show
    testStreamList= [RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", stream_name="TEST_0", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", stream_name="TEST_1", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", stream_name="TEST_2", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", stream_name="TEST_3", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", stream_name="TEST_4", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", stream_name="TEST_5", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", stream_name="TEST_6", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", stream_name="TEST_7", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", stream_name="TEST_8", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", stream_name="TEST_9", debug=debugMode),]

    testShowList=[]
    if showMode:
        for testStream in testStreamList:
            testShowList.append(show_imshow_demo(stream_queue=testStream.get_stream_queue()))


    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        for testShow in testShowList:
            del testShow
        for testStream in testStreamList:
            del testStream

        print("\nEND by KeyboardInterrupt")
        sys.exit(0)


import argparse
import sys
import time
import uuid
from multiprocessing import Manager
from threading import Thread

import av
import numpy as np

import demo_viewer
from dataclass_for_StreamFrameInstance import StreamFrameInstance



def _update_frame(rtsp_url, stream_name, stream_queue,debug=False, bypass_frame=5,):
    try:
        print(f"[INFO] RTSP URL: {rtsp_url} will OPEN")
        container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
        bypassed_count = 0
        for frame in container.decode(video=0):
                raw_stream_view = np.array(frame.to_ndarray(format='bgr24'))
                if debug: print(f"[{stream_name}] 수신: {raw_stream_view.shape}, 평균 밝기: {raw_stream_view.mean():.2f}")

                if bypassed_count < bypass_frame:
                    bypassed_count += 1
                    bypass_flag = True
                else:
                    bypass_flag = False
                    bypassed_count = 0

                stream_frame_instance = StreamFrameInstance(
                    stream_name=stream_name,
                    row_frame_bytes=raw_stream_view.tobytes(),
                    height=raw_stream_view.shape[0],
                    width=raw_stream_view.shape[1],
                    bypass_flag=bypass_flag,
                )

                if stream_queue.full():
                    stream_queue.get()
                stream_queue.put(stream_frame_instance)
                time.sleep(1/30)

    except Exception as e:
        print(f"[ERROR] {stream_name} 스레드 예외 발생: {e}")
        #container.close()


class RtspStream:
    def __init__(self, rtsp_url, manager_queue, stream_name=str(uuid.uuid4()), bypass_frame=0, debug=False,):
        self.rawStreamView = None
        self.rtsp_url = rtsp_url
        self.stream_name = stream_name
        self.debug=debug
        self.stream_queue = manager_queue
        self.bypass_frame = bypass_frame


        #self.stream_queue = manager_queue.Queue(maxsize=720)
        self.stream_thread = Thread(target=_update_frame, name=self.stream_name,
                                      args=(self.rtsp_url, self.stream_name, self.stream_queue, self.debug))
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def get_stream_name(self):
        return self.stream_name

    def get_stream_queue(self):
        return self.stream_queue

    def __del__(self):
        if self.stream_thread.is_alive():
            self.stream_queue.put(None)
            self.stream_thread.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="디폴트 값 테스트")
    parser.add_argument("--debug", type=bool, default=False, help="디버그 로그 출력")
    parser.add_argument("--show", type=bool, default=True, help="스트림 화면 출력")
    args = parser.parse_args()

    debugMode=args.debug
    showMode=args.show
    queue = Manager().Queue(maxsize=512)

    testStreamList= [RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=queue, stream_name="TEST_0", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", manager_queue=queue, stream_name="TEST_1", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", manager_queue=queue, stream_name="TEST_2", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", manager_queue=queue, stream_name="TEST_3", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", manager_queue=queue, stream_name="TEST_4", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", manager_queue=queue, stream_name="TEST_5", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", manager_queue=queue, stream_name="TEST_6", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", manager_queue=queue, stream_name="TEST_7", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", manager_queue=queue, stream_name="TEST_8", debug=debugMode),
                     RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", manager_queue=queue, stream_name="TEST_9", debug=debugMode),]

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


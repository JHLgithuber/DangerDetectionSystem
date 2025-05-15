import time
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Queue, freeze_support
#from queue import Queue
from stream_input import RtspStream
from demo_viewer import start_imshow_demo



def main(url_list, debug_mode=True, show_mode=True,):
    #초기화
    frame_smm = SharedMemoryManager()
    frame_smm.start(

    #스트림 입력
    )
    input_metadata_queue = Queue()
    stram_instance_list=[]
    for name, url in url_list:
        print(f"name: {name}, url: {url}")
        stram_instance_list.append(RtspStream(rtsp_url=url, manager_smm=frame_smm, metadata_queue=input_metadata_queue ,stream_name=name, debug=debug_mode))


    #스트림 출력
    output_metadata_queue = Queue()
    if show_mode:
        print(f"show_mode is {show_mode}")

        start_imshow_demo(stream_queue=output_metadata_queue, debug=debug_mode)
    while True:
        output_metadata_queue.put(input_metadata_queue.get())
        if debug_mode:
            print("main")
            print(f"input_metadata_queue.qsize(): {input_metadata_queue.qsize()}")
            print(f"output_metadata_queue.qsize(): {output_metadata_queue.qsize()}")
            print("--------------------------------")
            time.sleep(0.01)



if __name__ == "__main__":
    freeze_support()
    test_url_list = [
        ("TEST_0", "rtsp://210.99.70.120:1935/live/cctv068.stream"),
        #("TEST_1", "rtsp://210.99.70.120:1935/live/cctv069.stream"),
        #("TEST_2", "rtsp://210.99.70.120:1935/live/cctv070.stream"),
        #("TEST_3", "rtsp://210.99.70.120:1935/live/cctv071.stream"),
        #("TEST_4", "rtsp://210.99.70.120:1935/live/cctv072.stream"),
        #("TEST_5", "rtsp://210.99.70.120:1935/live/cctv073.stream"),
        #("TEST_6", "rtsp://210.99.70.120:1935/live/cctv074.stream"),
        #("TEST_7", "rtsp://210.99.70.120:1935/live/cctv075.stream"),
        #("TEST_8", "rtsp://210.99.70.120:1935/live/cctv076.stream"),
        #("TEST_9", "rtsp://210.99.70.120:1935/live/cctv077.stream"),
        ]
    main(test_url_list)
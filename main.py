import time
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Queue, freeze_support, set_start_method

import cv2

import dataclass_for_StreamFrameInstance
#from queue import Queue
from stream_input import RtspStream
from demo_viewer import start_imshow_demo



def main(url_list, debug_mode=True, show_mode=True, max_frames=1000):
    #입력 스트림 초기화
    input_metadata_queue = Queue()
    stram_instance_dict=dict()
    for name, url in url_list:
        print(f"name: {name}, url: {url}")
        stram_instance_dict[name]=RtspStream(rtsp_url=url, metadata_queue=input_metadata_queue ,stream_name=name, debug=debug_mode)


    #공유메모리 설정
    frame_smm_mgr = SharedMemoryManager()
    frame_smm_mgr.start()
    shm_objs_dict=dict()
    shm_names_dict=dict()
    for name, instance in stram_instance_dict.items():
        shm_objs=[frame_smm_mgr.SharedMemory(size=instance.get_bytes()) for _ in range(max_frames)]
        shm_name=[shm.name for shm in shm_objs]
        shm_objs_dict[name]=shm_objs
        shm_names_dict[name]=shm_name
        if debug_mode: print(shm_name)
        if debug_mode: print(shm_objs)

    #입력 스트림 실행
    for name, instance in stram_instance_dict.items():
        instance.run_stream(shm_names_dict[name])


    #출력 스트림 설정
    output_metadata_queue = Queue()
    start_imshow_demo(output_metadata_queue, debug=debug_mode)

    #뭔가 프로세싱
    while True:
        porc_frame=input_metadata_queue.get()
        time.sleep(0.01)
        if debug_mode: print(f"porc_frame.captured_datetime: {porc_frame.captured_datetime}")
        output_metadata_queue.put(porc_frame)





if __name__ == "__main__":
    set_start_method('spawn', force=True)
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
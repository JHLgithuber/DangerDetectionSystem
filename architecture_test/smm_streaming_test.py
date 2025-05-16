import time

import cv2
import numpy as np
from multiprocessing import Process, Semaphore, Value, Event
from multiprocessing.managers import SharedMemoryManager

BUFFER_SIZE = 100

def producer(sl, free_slots, stored_slots, head, stop_event, video_path, frame_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 빈 슬롯이 생길 때까지 대기
        free_slots.acquire()

        # head 인덱스에 바이트 스트림 저장
        with head.get_lock():
            idx = head.value % BUFFER_SIZE
            sl[idx] = frame.tobytes()
            head.value += 1

        # 슬롯 채워졌음을 알림
        stored_slots.release()

    cap.release()
    stop_event.set()

def consumer(sl, free_slots, stored_slots, tail, stop_event, h, w):
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    while True:
        # 읽을 슬롯이 생길 때까지 최대 1초 대기
        if not stored_slots.acquire(timeout=1):
            if stop_event.is_set():
                break
            continue

        time.sleep(1/60)
        # tail 인덱스에서 바이트 꺼내기
        with tail.get_lock():
            idx = tail.value % BUFFER_SIZE
            data = sl[idx]
            tail.value += 1

        # 슬롯 비워졌음을 알림
        free_slots.release()

        # bytes → NumPy 배열 → 화면 출력
        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../streetTestVideo.mp4"

    # 1) 해상도 얻기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    h, w = frame0.shape[:2]
    frame_size = h * w * 3  # bytes per frame

    # 2) SharedMemoryManager & ShareableList 생성
    smm = SharedMemoryManager()
    smm.start()
    # ShareableList 초기값: 모두 빈 바이트
    empty = b'\x00' * frame_size
    sl = smm.ShareableList([empty] * BUFFER_SIZE)

    # 3) 세마포어/인덱스/이벤트
    free_slots   = Semaphore(BUFFER_SIZE)  # 빈 슬롯 개수
    stored_slots = Semaphore(0)            # 채워진 슬롯 개수
    head  = Value('i', 0)                  # producer 쓰기 위치
    tail  = Value('i', 0)                  # consumer 읽기 위치
    stop_event   = Event()                 # 종료 신호

    # 4) 프로세스 생성
    p_prod = Process(
        target=producer,
        args=(sl, free_slots, stored_slots, head, stop_event, video_path, frame_size)
    )
    p_cons = Process(
        target=consumer,
        args=(sl, free_slots, stored_slots, tail, stop_event, h, w)
    )

    p_prod.start()
    p_cons.start()
    p_prod.join()
    p_cons.join()

    # 5) 정리
    # ShareableList 내부 SharedMemory 닫기/해제
    sl.shm.close()
    sl.shm.unlink()
    smm.shutdown()

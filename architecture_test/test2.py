import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

def producer(video_path, shm_names, queue, frame_shape, frame_nbytes):
    # SharedMemory 연결 & NumPy 뷰 생성
    shms = [SharedMemory(name=name) for name in shm_names]
    buffers = [np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf) for shm in shms]

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 크기가 다르면 리사이즈
        if frame.nbytes != frame_nbytes:
            frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
        # 버퍼에 복사
        np.copyto(buffers[idx], frame)
        # 사용 가능한 인덱스 전송
        queue.put(idx)
        idx = (idx + 1) % len(buffers)

    # 종료 신호
    queue.put(None)
    cap.release()

def consumer(shm_names, queue, frame_shape):
    shms = [SharedMemory(name=name) for name in shm_names]
    buffers = [np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf) for shm in shms]

    while True:
        idx = queue.get()
        if idx is None:        # 종료 신호
            break
        frame = buffers[idx]
        cv2.imshow("SharedMemory Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../streetTestVideo.mp4"   # 사용할 동영상 파일 경로
    max_frames = 100

    # --- 1) 첫 프레임으로 크기 확인 ---
    cap = cv2.VideoCapture(video_path)
    ret, first = cap.read()
    if not ret:
        print("영상을 로드할 수 없습니다.")
        exit(1)
    frame_shape = first.shape        # (height, width, 3)
    frame_nbytes = first.nbytes
    cap.release()

    # --- 2) SharedMemoryManager 시작 & 메모리 할당 ---
    mgr = SharedMemoryManager()
    mgr.start()
    shm_objs = [mgr.SharedMemory(size=frame_nbytes) for _ in range(max_frames)]
    shm_names = [shm.name for shm in shm_objs]
    print(type(shm_names[0]))

    # --- 3) 프로세스 및 큐 생성 ---
    queue = mp.Queue(maxsize=max_frames)
    p_prod = mp.Process(target=producer,
                        args=(video_path, shm_names, queue, frame_shape, frame_nbytes))
    p_cons = mp.Process(target=consumer,
                        args=(shm_names, queue, frame_shape))

    # --- 4) 실행 & 정리 ---
    p_prod.start()
    p_cons.start()
    p_prod.join()
    p_cons.join()

    mgr.shutdown()

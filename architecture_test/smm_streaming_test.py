import time
from multiprocessing import Process, Queue
from multiprocessing.managers import SharedMemoryManager
import numpy as np

def save_frame_to_shared_memory(frame: np.ndarray, smm: SharedMemoryManager):
    shm = smm.SharedMemory(size=frame.nbytes)
    shm_arr = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    shm_arr[:] = frame[:]
    return {"name": shm.name, "shape": frame.shape, "dtype": frame.dtype.str}

def load_frame_from_shared_memory(frame_info):
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=frame_info['name'])
    dtype = np.dtype(frame_info['dtype'])
    arr = np.ndarray(frame_info['shape'], dtype=dtype, buffer=shm.buf)
    frame = arr.copy()
    shm.close()
    return frame

def producer(smm, queue):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    shm_info = save_frame_to_shared_memory(frame, smm)
    print(f"[Producer] SharedMemory name: {shm_info['name']}")
    queue.put(shm_info)
    # 자식 프로세스가 충분히 접근할 때까지 대기
    time.sleep(3)

def consumer(queue):
    item = queue.get()
    print(f"[Consumer] Received frame info: {item}")
    frame = load_frame_from_shared_memory(item)
    print(f"[Consumer] Frame shape: {frame.shape}, mean: {frame.mean():.2f}")

if __name__ == "__main__":
    smm = SharedMemoryManager()
    smm.start()
    queue = Queue()

    p = Process(target=consumer, args=(queue,))
    p.start()

    producer(smm, queue)

    p.join()
    smm.shutdown()

from multiprocessing import shared_memory

import numpy as np
import cv2
TARGET_WIDTH = 960
TAREGET_HEIGHT= 540
TARGET_DEPTH = 3

def arr_steam_player(shm_name, shape=(TAREGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH)):
    # Shared Memory 테스트용, shared 메모리로부터 배열을 받아서,
    # print('[arr_stream] getting ')
    existing_shm = shared_memory.SharedMemory(name=shm_name)


    shared_a = np.frombuffer(existing_shm.buf, dtype=np.uint8)
    c = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)

    print(type(c))
    print(shared_a.shape, c.shape)

    while True:
        cv2.imshow('SharedMemory_player', c)
        if cv2.waitKey(int(1000 / 24)) == ord('q'):
            break

    existing_shm.unlink()
    existing_shm.close()
    # 영상 파일과 창 닫기
    cv2.destroyAllWindows()


if __name__ == '__main__':
    arr_steam_player('shm')
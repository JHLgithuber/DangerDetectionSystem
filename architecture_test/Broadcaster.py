from multiprocessing import shared_memory
import numpy as np
import cv2

TARGET_WIDTH = 960
TAREGET_HEIGHT= 540
TARGET_DEPTH = 3
video_path = '../data_for_test/streetTestVideo4.mp4'

sample_array = np.zeros((TAREGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH), dtype=np.uint8)  # (600768,)
stream_shm = shared_memory.SharedMemory(name ='shm',create=True, size=sample_array.nbytes)

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)

    # 영상 파일이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("영상 파일을 열 수 없습니다.")
        exit()

    # 영상의 프레임 너비와 높이 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력할 창 생성
    cv2.namedWindow('Broadcasting', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Broadcasting', frame_width, frame_height)

    while True:
        # 영상에서 프레임 읽기
        ret, frame = cap.read()

        # 프레임을 제대로 읽지 못한 경우 종료
        if not ret:
            break

        # 프레임 출력
        cv2.imshow('Broadcasting', frame)

        shared_a = np.ndarray(frame.shape, dtype=frame.dtype, buffer=stream_shm.buf)
        shared_a[:] = frame

        if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            cap.open(video_path)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

    stream_shm.unlink()
    stream_shm.close()
    cap.release()
    cv2.destroyAllWindows()
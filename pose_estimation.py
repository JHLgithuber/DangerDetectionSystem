import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#파일 위치 미리 지정
input_video_path = "streetTestVideo.mp4"
save_video_path = 'in_theft_output.mp4'

cap = cv2.VideoCapture(input_video_path)

#재생할 파일의 넓이와 높이
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#video controller
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (int(width), int(height)))


with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 웹캠을 불러올 경우는 'continue', 동영상을 불러올 경우 'break'를 사용합니다.
            break

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', image) #코랩에서 돌릴거면 imshow()문은 주석처리할 것.
        out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
out.release()
cv2.destroyAllWindows()
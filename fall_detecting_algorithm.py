from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math

def detect_fall_angle(lm2d_list, torso_thresh=50, thigh_thresh=50, calf_thresh=50):
    # Mediapipe 랜드마크 좌표 (정규화 좌표)
    left_shoulder = lm2d_list[11]
    right_shoulder = lm2d_list[12]
    left_hip = lm2d_list[23]
    right_hip = lm2d_list[24]
    left_knee = lm2d_list[25]
    right_knee = lm2d_list[26]
    left_ankle = lm2d_list[27]
    right_ankle = lm2d_list[28]

    # 부위별 중앙점 계산
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    mid_hip_x = (left_hip.x + right_hip.x) / 2
    mid_hip_y = (left_hip.y + right_hip.y) / 2
    mid_knee_x = (left_knee.x + right_knee.x) / 2
    mid_knee_y = (left_knee.y + right_knee.y) / 2
    mid_ankle_x = (left_ankle.x + right_ankle.x) / 2
    mid_ankle_y = (left_ankle.y + right_ankle.y) / 2

    # 부위별 벡터 계산
    torso_vec = (mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y)
    thigh_vec = (mid_knee_x - mid_hip_x, mid_knee_y - mid_hip_y)
    calf_vec = (mid_ankle_x - mid_knee_x, mid_ankle_y - mid_knee_y)

    # 수직 방향(0,1)과 각 세그먼트 벡터의 이루는 각도 계산
    def vertical_angle(vec):
        # 수직 방향 (y축 아래방향)
        dot = vec[1] / (math.hypot(vec[0], vec[1]) + 1e-6)
        angle_deg = math.degrees(math.acos(min(max(dot, -1.0), 1.0)))
        return angle_deg

    angle_torso = vertical_angle(torso_vec)
    angle_thigh = vertical_angle(thigh_vec)
    angle_calf = vertical_angle(calf_vec)

    # 각도 출력 (디버깅 용도)
    print(f"Torso Angle: {angle_torso:.2f}°, Thigh Angle: {angle_thigh:.2f}°, Calf Angle: {angle_calf:.2f}°")

    # 쓰러짐 조건
    fallen_torso = angle_torso > torso_thresh
    fallen_thigh = angle_thigh > thigh_thresh
    fallen_calf = angle_calf > calf_thresh

    # 최종 판단 (세 부분 중 두 부분 이상 넘어지면 쓰러짐)
    fallen_parts = sum([fallen_torso, fallen_thigh, fallen_calf])
    is_fallen = fallen_parts >= 2

    # 방향 판단
    fall_direction = None
    if is_fallen:
        if torso_vec[0] > 0.1:
            fall_direction = 'right'
        elif torso_vec[0] < -0.1:
            fall_direction = 'left'
        else:
            if torso_vec[1] > 0:
                fall_direction = 'forward'
            else:
                fall_direction = 'backward'


    return (is_fallen, fall_direction)


def detect_fall_normalized(lm2d_list):
    # 2D 정규화 좌표 기반으로 모든 계산
    left_shoulder = lm2d_list[11]
    right_shoulder = lm2d_list[12]
    left_hip = lm2d_list[23]
    right_hip = lm2d_list[24]
    left_knee = lm2d_list[25]
    right_knee = lm2d_list[26]
    left_ankle = lm2d_list[27]
    right_ankle = lm2d_list[28]

    # 중앙점 계산 (정규화 좌표)
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    mid_hip_x = (left_hip.x + right_hip.x) / 2.0
    mid_hip_y = (left_hip.y + right_hip.y) / 2.0
    mid_knee_x = (left_knee.x + right_knee.x) / 2.0
    mid_knee_y = (left_knee.y + right_knee.y) / 2.0
    mid_ankle_x = (left_ankle.x + right_ankle.x) / 2.0
    mid_ankle_y = (left_ankle.y + right_ankle.y) / 2.0

    # 척추 벡터 및 길이 (정규화)
    spine_vec_x = mid_hip_x - mid_shoulder_x
    spine_vec_y = mid_hip_y - mid_shoulder_y
    spine_length = (spine_vec_x ** 2 + spine_vec_y ** 2) ** 0.5

    # 허리 폭 (정규화)
    waist_vec_x = right_hip.x - left_hip.x
    waist_vec_y = right_hip.y - left_hip.y
    waist_width = (waist_vec_x ** 2 + waist_vec_y ** 2) ** 0.5

    # 척추 대 허리 비율
    spine_ratio = spine_length / waist_width if waist_width > 1e-6 else float('inf')

    # 척추 각도 (정규화 좌표에서도 방식 동일)
    dot = spine_vec_y / (spine_length + 1e-6)  # 중력방향이 (0,1)이라서
    dot = max(min(dot, 1), -1)
    spine_angle_deg = math.degrees(math.acos(dot))

    # 좌우 어깨/엉덩이 높이 차이(정규화)
    shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
    hip_y_diff = abs(left_hip.y - right_hip.y)

    # 임계값은 정규화 단위(대략 0.18~0.2, 실험적으로 조정)
    is_side_fall = (shoulder_y_diff > 0.18) or (hip_y_diff > 0.18)
    is_fall = (spine_angle_deg > 50) or (spine_ratio < 1.2)
    is_fall_final = is_fall or is_side_fall

    # 방향 판정
    fall_direction = None
    if is_fall_final:
        if is_side_fall:
            if left_shoulder.y > right_shoulder.y:
                fall_direction = 'left'
            else:
                fall_direction = 'right'
        else:
            if mid_shoulder_y - mid_hip_y > 0:
                fall_direction = 'forward'
            elif mid_ankle_y - mid_hip_y < 0:
                fall_direction = 'backward'
            else:
                fall_direction = None

    return (is_fall_final, fall_direction)

#신뢰도 검사
def check_visibility_presence(lm2d_list):
    check_point_list=list()
    check_point_list.append(lm2d_list[11])  #left_shoulder
    check_point_list.append(lm2d_list[12])  #right_shoulder
    check_point_list.append(lm2d_list[23])  #left_hip
    check_point_list.append(lm2d_list[24])  #right_hip
    check_point_list.append(lm2d_list[25])  #left_knee
    check_point_list.append(lm2d_list[26])  #right_knee
    check_point_list.append(lm2d_list[27])  #left_ankle
    check_point_list.append(lm2d_list[28])  #right_ankle
    for lm in check_point_list:
        if not (hasattr(lm, 'visibility') and lm.visibility > 0.00 and hasattr(lm, 'presence') and lm.presence > 0.6):
            return False
    return True

def detect_fall(detection_result, debug=False):
    # 2D 정규화 랜드마크 리스트 (list of list)
    pixel_landmarks_list = detection_result.pose_landmarks

    # 둘 중 하나라도 없으면 원본 반환
    if not pixel_landmarks_list:
        return None

    # 인물(사람)별로 순회
    for idx in range(len(pixel_landmarks_list)):
        lm2d_list = pixel_landmarks_list[idx]
        if not lm2d_list:
            continue
        else:
            if not check_visibility_presence(lm2d_list):
                return None


        #result = detect_fall_normalized(lm2d_list)
        result = detect_fall_angle(lm2d_list)
        if result[0]:  # is_fall이 True인 경우
            if result[1] == 'forward':
                print("낙상 감지: 앞으로 넘어짐")
            elif result[1] == 'backward':
                print("낙상 감지: 뒤로 넘어짐")
            else:
                print("낙상 감지: 깊이 방향으로 넘어짐 (방향 불명)")
        else:
            print("넘어짐 없음: 정상 자세")

        if result[0]:
            if debug: print("FALL DETECTED")
            return True
        else:
            if debug: print("FALL NOT DETECTED")
            return False

    return None


if __name__ == "__main__":
    left_ankle_lm = landmark_pb2.Landmark(x=0.2, y=0, z=0.3, presence=1.0)
    right_ankle_lm = landmark_pb2.Landmark(x=-0.8, y=0, z=-0.3, presence=1.0)

    # _is_center_in_ellipse(left_ankle_lm, right_ankle_lm, margin_ratio=1.2, debug=True)

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import numpy as np

def _is_sitting_pose(joint_list, y_threshold=0.15, knee_angle_threshold=120, presence=0.6, debug=False):
    """
    착석 자세 판별 (3D 월드 랜드마크 리스트 기반)
    - y_threshold: 힙/무릎, 무릎/발목 간 y차이가 이 값 이하면 '수평'
    - knee_angle_threshold: 무릎 각도가 이 값 이하이면 '앉음' (deg)
    """
    for joint in joint_list:
        if joint.presence <presence:
            if debug: print("_is_sitting_pose: presence is to LOW")
            return None

    # 주요 관절
    left_hip = joint_list[0]
    right_hip = joint_list[1]
    left_knee = joint_list[2]
    right_knee = joint_list[3]
    left_ankle = joint_list[4]
    right_ankle = joint_list[5]

    # 각 관절의 y(높이)
    hip_y = (left_hip.y + right_hip.y) / 2
    knee_y = (left_knee.y + right_knee.y) / 2
    ankle_y = (left_ankle.y + right_ankle.y) / 2

    # 1. 엉덩이/무릎, 무릎/발목 간 y값 비교 (y값이 클수록 바닥과 멀어짐)
    # 앉음: 힙과 무릎이 거의 같은 높이, 무릎이 발목보다 높음
    is_hip_near_knee = abs(hip_y - knee_y) < y_threshold
    is_knee_above_ankle = (knee_y < ankle_y)  # mediapipe 기준 y축 방향에 따라 부호 주의!

    # 2. 무릎 각도 (벡터 사이 각도)
    def angle_between(a, b, c):
        # 각도 (deg) for angle at b
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        return np.degrees(theta)

    left_knee_angle = angle_between(left_hip, left_knee, left_ankle)
    right_knee_angle = angle_between(right_hip, right_knee, right_ankle)
    mean_knee_angle = (left_knee_angle + right_knee_angle) / 2

    # 3. 최종 판별
    # 'hip이 knee와 비슷하고', 'knee가 ankle보다 위', 'knee각도가 threshold 이하'면 앉음
    sitting = is_hip_near_knee and is_knee_above_ankle and (mean_knee_angle < knee_angle_threshold)

    # (디버깅용)
    if debug:
        print(f"[SITTING DETECT] hip_y={hip_y:.3f}, knee_y={knee_y:.3f}, ankle_y={ankle_y:.3f}")
        print(f"   is_hip_near_knee={is_hip_near_knee}, is_knee_above_ankle={is_knee_above_ankle}")
        print(f"   left_knee_angle={left_knee_angle:.1f}, right_knee_angle={right_knee_angle:.1f}, mean={mean_knee_angle:.1f}")
        print(f"   -> sitting? {sitting}")

    return sitting


def _is_center_in_ellipse(left_ankle_landmark, right_ankle_landmark, presence=0.6, margin=1.5, debug=False):

    if left_ankle_landmark.presence <presence or right_ankle_landmark.presence < presence:
        if debug: print("_is_center_in_ellipse: presence is to LOW")
        return None

    left_ankle_point = np.array([left_ankle_landmark.x, left_ankle_landmark.z])
    right_ankle_point = np.array([right_ankle_landmark.x, right_ankle_landmark.z])

    # 1. 두 초점 간 거리
    focal_dist = np.linalg.norm(left_ankle_point - right_ankle_point)

    # 2. 장축의 길이(2a): 초점 거리 * margin_ratio(여유분, 실제 사람의 경우 10~20cm 정도)
    two_a = focal_dist + margin

    # 3. 타원의 중심(엉덩이 평균)은 (0, 0)으로 가정
    P = np.array([0.0, 0.0])

    # 4. 두 초점까지 거리의 합
    d_sum = np.linalg.norm(P - left_ankle_point) + np.linalg.norm(P - right_ankle_point)

    # 5. 판정
    in_ellipse = d_sum <= two_a

    if debug:
        # 결과 출력
        print(f"두 초점 거리: {focal_dist:.4f}")
        print(f"장축 2a: {two_a:.4f}")
        print(f"(0,0)에서 두 초점까지 거리 합: {d_sum:.4f}")
        print("포함여부:", in_ellipse)

    return in_ellipse

def detect_fall(detection_result, debug=False):
    # 2D 정규화 랜드마크 리스트 (list of list)
    pixel_landmarks_list = detection_result.pose_landmarks
    # 3D 월드 랜드마크 리스트 (list of list)
    world_landmarks_list = detection_result.pose_world_landmarks

    # 둘 중 하나라도 없으면 원본 반환
    if not pixel_landmarks_list or not world_landmarks_list:
        return None

    # 인물(사람)별로 순회
    for idx in range(len(world_landmarks_list)):
        lm3d_list = world_landmarks_list[idx]
        if not lm3d_list:
            continue

        #무게중심 추출에 사용, 근데 평균 0 아니니?
        left_hip_landmark = lm3d_list[23]
        right_hip_landmark = lm3d_list[24]

        #앉은건지, 넘어진건지 판단용
        left_knee_landmark = lm3d_list[25]
        right_knee_landmark = lm3d_list[26]

        #받침점 추출에 사용
        left_ankle_landmark = lm3d_list[27]
        right_ankle_landmark = lm3d_list[28]

        #정밀 탐색용? 이렇게 정밀할 필요가 있을까
        left_heel_landmark = lm3d_list[29]
        right_heel_landmark = lm3d_list[30]
        left_foot_index_landmark = lm3d_list[31]
        right_foot_index_landmark = lm3d_list[32]


        is_center_of_mass_in_support_area=_is_center_in_ellipse(left_ankle_landmark, right_ankle_landmark,
                                                                presence=0.7, margin=2, debug=debug)
        if is_center_of_mass_in_support_area is None:
            return None

        #is_not_sitting=not(_is_sitting_pose([left_hip_landmark, right_hip_landmark,
        #                             left_knee_landmark, right_knee_landmark,
        #                             left_ankle_landmark, right_ankle_landmark],
        #                            y_threshold=0.15, knee_angle_threshold=120, presence=0.7, debug=debug
        #                            ))
        is_not_sitting=True

        if debug:
            print(f"is_center_of_mass_in_support_area: {is_center_of_mass_in_support_area}")
            print(f"is_not_sitting: {is_not_sitting}")


        if is_center_of_mass_in_support_area and is_not_sitting:
            if debug: print("FALL DETECTED")
            return True
        else:
            if debug: print("FALL NOT DETECTED")
            return False

    return None

if __name__ == "__main__":
    left_ankle_lm = landmark_pb2.Landmark(x=0.2, y=0, z=0.3, presence=1.0)
    right_ankle_lm = landmark_pb2.Landmark(x=-0.8, y=0, z=-0.3, presence=1.0)

    _is_center_in_ellipse(left_ankle_lm, right_ankle_lm, margin_ratio=1.2, debug=True)
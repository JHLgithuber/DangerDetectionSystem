import math


def detect_fall_static_spine_angle(lm2d_list, torso_thresh=50):
    """
    척추 기울기 기반 낙상 감지

    원리:
        - 좌우 어깨의 중간점과 좌우 골반의 중간점을 이은 벡터(척추)의 기울기를 계산함
        - 해당 벡터가 수직(y축)과 이루는 각도가 일정 기준 이상이면, 넘어졌다고 판단

    민감도 조정 방법:
        - torso_thresh 값을 낮추면 더 작은 기울기에서도 낙상으로 판단하므로 민감도가 높아짐

    Args:
        lm2d_list (list): (17,2) 형태의 랜드마크 리스트
        torso_thresh (float): 기울기 각도 임계값 (deg)

    Returns:
        tuple(bool, float): 낙상 여부, 기울기 각도
    """
    left_shoulder = lm2d_list[5]
    right_shoulder = lm2d_list[6]
    left_hip = lm2d_list[11]
    right_hip = lm2d_list[12]

    mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2]
    mid_hip = [(left_hip[0] + right_hip[0]) / 2,
               (left_hip[1] + right_hip[1]) / 2]

    vec = [mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1]]
    dot = vec[1] / (math.hypot(vec[0], vec[1]) + 1e-6)
    angle = math.degrees(math.acos(max(min(dot, 1), -1)))
    return angle > torso_thresh, angle


def detect_fall_static_shoulder_hip_diff(lm2d_list, diff_thresh=0.18):
    """
    어깨/엉덩이 높이 차이 기반 낙상 감지

    원리:
        - 사람은 넘어지면 한쪽이 더 낮아지는 비대칭 자세를 취함
        - 좌우 어깨나 엉덩이의 높이 차이가 일정 이상이면 낙상으로 판단

    민감도 조정 방법:
        - diff_thresh 값을 낮추면 작은 비대칭에도 민감하게 반응

    Returns:
        tuple(bool, tuple): 낙상 여부, (어깨 높이차, 엉덩이 높이차)
    """
    shoulder_y_diff = abs(lm2d_list[5][1] - lm2d_list[6][1])
    hip_y_diff = abs(lm2d_list[11][1] - lm2d_list[12][1])
    return (shoulder_y_diff > diff_thresh or hip_y_diff > diff_thresh), (shoulder_y_diff, hip_y_diff)


def detect_fall_static_recline_ratio(lm2d_list, min_recline_ratio=0.9):
    """
    세로/가로 비율 기반 낙상 감지

    원리:
        - 사람이 서 있을 땐 세로로 길고, 누우면 가로로 퍼진다
        - 어깨~발 길이 / 어깨 간 거리 비율이 작으면 낙상으로 판단

    민감도 조정 방법:
        - min_recline_ratio 값을 높이면 더 많은 자세가 낙상으로 간주됨

    Returns:
        tuple(bool, float): 낙상 여부, 세로:가로 비율
    """
    left_shoulder = lm2d_list[5]
    right_shoulder = lm2d_list[6]
    left_ankle = lm2d_list[15]
    right_ankle = lm2d_list[16]

    width = abs(right_shoulder[0] - left_shoulder[0])
    height = abs((right_shoulder[1] - right_ankle[1] + left_shoulder[1] - left_ankle[1]) / 2.0)

    ratio = height / (width + 1e-6)
    return ratio < min_recline_ratio, ratio


def detect_fall_static_joint_bbox_ratio(lm2d_list, max_ratio=2.0):
    """
    전체 관절의 바운딩 박스 비율 기반 낙상 감지 (팔 제외)

    원리:
        - 팔(팔꿈치, 손목)을 제외한 주요 관절만으로 바운딩 박스를 생성
        - 누운 자세일수록 세로 길이가 짧아지고 가로로 퍼짐 → 비율 작아짐

    민감도 조정 방법:
        - max_ratio 값을 높이면 더 많은 가로 자세를 낙상으로 판단함

    Returns:
        tuple(bool, float): 낙상 여부, 바운딩 박스 세로:가로 비율
    """
    # 제외할 인덱스: 양팔 팔꿈치(7,8), 손목(9,10)
    exclude_indices = {7, 8, 9, 10}
    points = [pt for i, pt in enumerate(lm2d_list) if i not in exclude_indices]

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    ratio = h / (w + 1e-6)
    return ratio < max_ratio, ratio


def detect_fall_angle(lm2d_list, torso_thresh=50, thigh_thresh=50, calf_thresh=50, leg_thresh=50, debug=False, ):
    """
    2D 포즈 각도 기반 낙상 여부 판단

    Args:
        lm2d_list (list): Mediapipe 2D 랜드마크 리스트
        torso_thresh (float): 상체 기울기 임계각 (deg)
        thigh_thresh (float): 허벅지 기울기 임계각 (deg)
        calf_thresh (float): 종아리 기울기 임계각 (deg)
        leg_thresh (float): 좌/우 종아리 개별 임계각 (deg)
        debug (bool): 디버그용 출력 여부

    Returns:
        tuple:
            - is_fallen (bool): 낙상 판단 결과
            - fallen_reason (str): 낙상 판정 이유 요약 문자열
    """
    left_shoulder = lm2d_list[5]
    right_shoulder = lm2d_list[6]
    left_hip = lm2d_list[11]
    right_hip = lm2d_list[12]
    left_knee = lm2d_list[13]
    right_knee = lm2d_list[14]
    left_ankle = lm2d_list[15]
    right_ankle = lm2d_list[16]

    # 부위별 중앙점 계산
    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    mid_hip_x = (left_hip[0] + right_hip[0]) / 2
    mid_hip_y = (left_hip[1] + right_hip[1]) / 2
    mid_knee_x = (left_knee[0] + right_knee[0]) / 2
    mid_knee_y = (left_knee[1] + right_knee[1]) / 2
    mid_ankle_x = (left_ankle[0] + right_ankle[0]) / 2
    mid_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    # 부위별 벡터 계산
    torso_vec = (mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y)
    thigh_vec = (mid_knee_x - mid_hip_x, mid_knee_y - mid_hip_y)
    calf_vec = (mid_ankle_x - mid_knee_x, mid_ankle_y - mid_knee_y)
    left_calf_vec = (left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1])
    right_calf_vec = (right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1])

    if debug:
        print(
            f"[FALL_ANGLE] torso_vec: {torso_vec}, thigh_vec: {thigh_vec}, calf_vec: {calf_vec}, left_calf_vec: {left_calf_vec}, right_calf_vec: {right_calf_vec}")

    # 수직 방향(0,1)과 각 segment 벡터의 이루는 각도 계산
    def vertical_angle(vec):
        # 수직 방향 (y축 아래 방향)
        dot = vec[1] / (math.hypot(vec[0], vec[1]) + 1e-6)
        angle_deg = math.degrees(math.acos(min(max(dot, -1.0), 1.0)))
        return angle_deg

    # 부위별 각도 계산
    angle_torso = vertical_angle(torso_vec)
    angle_thigh = vertical_angle(thigh_vec)
    angle_calf = vertical_angle(calf_vec)
    angle_left_calf = vertical_angle(left_calf_vec)
    angle_right_calf = vertical_angle(right_calf_vec)

    # 쓰러짐 조건
    fallen_torso = angle_torso > torso_thresh
    fallen_thigh = angle_thigh > thigh_thresh
    fallen_calf = angle_calf > calf_thresh
    fallen_left_calf = angle_left_calf > leg_thresh
    fallen_right_calf = angle_right_calf > leg_thresh

    # 최종 판단 (세 부분 중 일정 부분 이상 넘어 지면 쓰러짐)
    fallen_reason = f"Torso: {fallen_torso}({angle_torso}) | Thigh: {fallen_thigh}({angle_thigh}) | Calf: {fallen_calf}({angle_calf}) | Left calf: {fallen_left_calf}({angle_left_calf}) | Right calf: {fallen_right_calf}({angle_right_calf})"
    is_fallen = False
    if fallen_torso:
        fallen_parts = sum([fallen_torso, fallen_thigh, fallen_calf, fallen_left_calf, fallen_right_calf, ])
        is_fallen = fallen_parts >= 4

    return is_fallen, fallen_reason


# noinspection PyUnusedLocal
def detect_fall_recline(lm2d_list, min_recline_ratio=2.0, debug=False):
    """
    신체 세로:가로 비율이 낮으면 누운 상태로 판정
    min_recline_ratio는 영상에서 보이는 정상적인 신체비율로 설정, 폭 대비 키가 짧아져 보이면 트리거
    Args:
        lm2d_list (list): Mediapipe 2D 랜드마크 리스트
        min_recline_ratio (float): 넘어짐 판단 기준 세로:가로 비율 임계값
        debug (bool): 디버그 출력 여부

    Returns:
        tuple:
            - is_fallen (bool): 낙상 여부
            - fallen_reason (str): 비율 비교에 따른 결과 설명
    """
    left_shoulder = lm2d_list[5]
    right_shoulder = lm2d_list[6]
    left_hip = lm2d_list[11]
    right_hip = lm2d_list[12]
    left_knee = lm2d_list[13]
    right_knee = lm2d_list[14]
    left_ankle = lm2d_list[15]
    right_ankle = lm2d_list[16]

    # 신체 비율 계산
    width_body = abs(right_shoulder[0] - left_shoulder[0])
    height_body = abs((right_shoulder[1] - right_ankle[1] + left_shoulder[1] - left_ankle[1]) / 2.0)
    ratio_body = height_body / width_body if width_body > 1e-6 else float('inf')

    if ratio_body < min_recline_ratio:
        is_fallen = True
        fallen_reason = f"ratio_body({ratio_body}) < min_recline_ratio({min_recline_ratio})"
    else:
        is_fallen = False
        fallen_reason = f"ratio_body({ratio_body}) in range({min_recline_ratio})"

    return is_fallen, fallen_reason


def check_visibility_presence(conf_array, threshold=0.5):
    """
    YOLOv11 포즈 키포인트 신뢰도 검사

    Args:
        conf_array (array-like): (17,) 형태의 confidence 배열
        threshold (float): 신뢰도 기준값

    Returns:
        bool: 주요 관절의 confidence가 기준 이상일 경우 True
    """
    # YOLOv11 (COCO 포맷) 기준 주요 관절 인덱스
    keypoint_indices = [5, 6, 11, 12, 13, 14, 15, 16]

    # 배열 타입과 길이 확인
    if conf_array is None or len(conf_array) < max(keypoint_indices) + 1:
        return False

    for idx in keypoint_indices:
        if conf_array[idx] < threshold:
            return False
    return True


def detect_fall(detection_result, conf, debug=False):
    """
    여러 낙상 판단 알고리즘 기반 낙상 감지

    Args:
        detection_result: pose_landmarks 속성을 포함한 객체 (사람별 랜드마크 리스트)
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        bool or None:
            - True: 낙상 감지됨
            - False: 낙상 아님
            - None: 판단 불가 (신뢰도 부족 또는 landmark 없음)
    """

    # 랜드마크 리스트가 없으면 반환
    if detection_result is None or len(detection_result) == 0:
        return None

    if not check_visibility_presence(conf):
        return None

    result_by_angle = detect_fall_static_spine_angle(detection_result)
    result_by_recline = detect_fall_static_recline_ratio(detection_result)
    result_by_bbox = detect_fall_static_joint_bbox_ratio(detection_result)
    result_by_asymmetry = detect_fall_static_shoulder_hip_diff(detection_result)

    if result_by_angle[0]:
        if debug: print(f"FALL DETECTED by spine angle: {result_by_angle[1]}")
        return True
    elif result_by_bbox[0]:
        if debug: print(f"FALL DETECTED by joint bbox ratio: {result_by_bbox[1]}")
        return True
    # elif result_by_asymmetry[0]:
    #    if debug: print(f"FALL DETECTED by shoulder/hip asymmetry: {result_by_asymmetry[1]}")
    #    return True
    else:
        if debug: print("FALL NOT DETECTED")
        return False

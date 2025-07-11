import math


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
    # Mediapipe landmark 좌표 (정규화 좌표)
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
    left_calf_vec = (left_ankle.x - left_knee.x, left_ankle.y - left_knee.y)
    right_calf_vec = (right_ankle.x - right_knee.x, right_ankle.y - right_knee.y)

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


def detect_fall_feet(lm2d_list, threshold=0.03, debug=False):
    """
    발끝 또는 뒤꿈치가 발목보다 위에 있으면 낙상으로 판정

    Args:
        lm2d_list (list): Mediapipe 2D 랜드마크 리스트
        threshold (float): y좌표 차이 임계값 (기본 0.03)
        debug (bool): 디버그 출력 여부

    Returns:
        tuple:
            - is_fallen (bool): 낙상 여부
            - reason (str): 판단 근거 문자열
    """
    left_ankle = lm2d_list[27]
    right_ankle = lm2d_list[28]
    left_toe = lm2d_list[31]
    right_toe = lm2d_list[32]
    left_heel = lm2d_list[29]
    right_heel = lm2d_list[30]

    # 뒤로 넘어짐
    left_toe_diff = left_ankle.y - left_toe.y
    right_toe_diff = right_ankle.y - right_toe.y
    fallen_back = (left_toe_diff > threshold) or (right_toe_diff > threshold)

    # 앞으로 엎어짐
    left_heel_diff = left_ankle.y - left_heel.y
    right_heel_diff = right_ankle.y - right_heel.y
    fallen_front = (left_heel_diff > threshold) or (right_heel_diff > threshold)

    is_fallen = fallen_back or fallen_front

    reason = (
        f"ToeDiff L:{left_toe_diff:.3f} R:{right_toe_diff:.3f} | "
        f"HeelDiff L:{left_heel_diff:.3f} R:{right_heel_diff:.3f} | "
        f"FallenBack: {fallen_back}, FallenFront: {fallen_front}"
    )

    if debug:
        print(f"[FALL_FEET] {reason}")

    return is_fallen, reason


# noinspection PyUnusedLocal
def detect_fall_recline(lm2d_list, min_recline_ratio=1.1, debug=False):
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
    left_shoulder = lm2d_list[11]
    right_shoulder = lm2d_list[12]
    left_hip = lm2d_list[23]
    right_hip = lm2d_list[24]
    left_knee = lm2d_list[25]
    right_knee = lm2d_list[26]
    left_ankle = lm2d_list[27]
    right_ankle = lm2d_list[28]

    # 신체 비율 계산
    width_body = abs(right_shoulder.x - left_shoulder.x)
    height_body = abs((right_shoulder.y - right_ankle.y + left_shoulder.y - left_ankle.y) / 2.0)
    ratio_body = height_body / width_body if width_body > 1e-6 else float('inf')

    if ratio_body < min_recline_ratio:
        is_fallen = True
        fallen_reason = f"ratio_body({ratio_body}) < min_recline_ratio({min_recline_ratio})"
    else:
        is_fallen = False
        fallen_reason = f"ratio_body({ratio_body}) in range({min_recline_ratio})"

    return is_fallen, fallen_reason


# noinspection PyUnusedLocal
def detect_fall_normalized(lm2d_list):
    """
    정규화 좌표 기반 낙상 감지 (척추 각도, 비율, 좌우 비대칭 포함)

    Args:
        lm2d_list (list): Mediapipe 2D 랜드마크 리스트

    Returns:
        tuple:
            - is_fall_final (bool): 낙상 판단 결과
            - reason (None): 설명 문자열 (미구현, None 반환)
    """
    # 2D 정규화 좌표 기반 으로 모든 계산
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

    # 척추 각도 (정규화 좌표 에서도 방식 동일)
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

    return is_fall_final, None


def check_visibility_presence(lm2d_list):
    """
    주요 관절의 visibility, presence 신뢰도 검사

    Args:
        lm2d_list (list): Mediapipe 2D 랜드마크 리스트

    Returns:
        bool: 모든 관절이 기준 이상일 경우 True, 하나라도 부족하면 False
    """
    check_point_list = list()
    check_point_list.append(lm2d_list[11])  # left_shoulder
    check_point_list.append(lm2d_list[12])  # right_shoulder
    check_point_list.append(lm2d_list[23])  # left_hip
    check_point_list.append(lm2d_list[24])  # right_hip
    check_point_list.append(lm2d_list[25])  # left_knee
    check_point_list.append(lm2d_list[26])  # right_knee
    check_point_list.append(lm2d_list[27])  # left_ankle
    check_point_list.append(lm2d_list[28])  # right_ankle
    for lm in check_point_list:
        if not (hasattr(lm, 'visibility') and lm.visibility > 0.00 and hasattr(lm, 'presence') and lm.presence > 0.6):
            return False
    return True


def detect_fall(detection_result, debug=False):
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
    # 2D 정규화 랜드마크 리스트 (list of list)
    pixel_landmarks_list = detection_result.pose_landmarks

    # 랜드마크 리스트가 없으면 반환
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

        result_by_normalization = detect_fall_normalized(lm2d_list)
        result_by_angle = detect_fall_angle(lm2d_list)
        result_by_recline = detect_fall_recline(lm2d_list)
        result_by_feet = detect_fall_feet(lm2d_list)

        if result_by_angle[0]:
            if debug: print(f"FALL DETECTED by angle: {result_by_angle[1]}")
            return True
        elif result_by_feet[0]:
            if debug: print(f"FALL DETECTED by feet: {result_by_feet[1]}")
            return True
        # elif result_by_recline[0]:
        #     if debug: print(f"FALL DETECTED by recline: {result_by_recline[1]}")
        #     return True
        # elif result_by_normalization[0]:
        #     if debug: print(f"FALL DETECTED by normalization: {result_by_normalization[1]}")
        #     return True
        else:
            if debug: print("FALL NOT DETECTED")
            return False

    return None

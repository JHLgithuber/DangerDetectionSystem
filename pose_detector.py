from multiprocessing import Process, current_process

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import dataclass_for_StreamFrameInstance
import fall_detecting_algorithm


# TODO: mediapipe가 GPU를 사용하도록 할 수 있을 텐데

def crop_objects(stream_frame_instance, padding=10, cls_conf=0.35, need_frame=True, debug=False):
    """
    stream_frame_instance 에서 감지된 객체들을 padding 만큼 여백을 두고 잘라내어 반환

    Args:
        stream_frame_instance (StreamFrameInstance): 입력 프레임 객체
        padding (int): 바운딩박스 여백 (픽셀 단위)
        cls_conf (float): confidence 임계값 이하 객체는 무시
        need_frame (bool): crop 이미지가 실제로 필요한 경우 True
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        list of dict: 각 객체 정보
            - 'crop': 잘라낸 이미지 (or None)
            - 'img_size': (h, w)
            - 'class': 클래스 ID 또는 이름
            - 'score': confidence 점수
            - 'bbox': (x1, y1, x2, y2) with padding
            - 'track_id': 트래킹 ID 또는 None
    """
    h, w = stream_frame_instance.height, stream_frame_instance.width
    frame = None
    # shm = None
    if need_frame:
        # 1) 원본 프레임 복원
        frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
            stream_frame_instance, debug=debug, copy=True
        )

    crops = []

    # 2) Tracking 결과가 있으면 우선 처리
    if hasattr(stream_frame_instance, 'human_tracking_serial') and stream_frame_instance.human_tracking_serial:
        for obj in stream_frame_instance.human_tracking_serial:
            score = obj["confidence"]
            if score < cls_conf:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            # 여백 적용 및 경계값 클리핑
            x1_p = max(x1 - padding, 0)
            y1_p = max(y1 - padding, 0)
            x2_p = min(x2 + padding, w)
            y2_p = min(y2 + padding, h)
            crop = frame[y1_p:y2_p, x1_p:x2_p]
            crop = np.asarray(crop, dtype=np.uint8)
            crops.append({
                'crop': crop,
                'img_size': (y2_p - y1_p, x2_p - x1_p),  # (h, w)
                'class': obj["class"],
                'score': score,
                'bbox': (x1_p, y1_p, x2_p, y2_p),
                'track_id': obj["track_id"]
            })

    # 3) Detection(Numpy) 결과 처리
    elif hasattr(stream_frame_instance,
                 'human_detection_numpy') and stream_frame_instance.human_detection_numpy is not None:
        # detection 된 바운딩박스는 모델 입력 크기(test_size)에 맞춰져 있으므로 원본 크기로 스케일 복원
        test_size = (stream_frame_instance.human_detection_tsize,
                     stream_frame_instance.human_detection_tsize)
        ratio = min(test_size[0] / h, test_size[1] / w)

        output = stream_frame_instance.human_detection_numpy
        if output.numel() == 0:
            # if shm: shm.close()
            return crops

        bboxes = output[:, 0:4] / ratio  # [x1, y1, x2, y2]
        classes = output[:, 6]
        scores = output[:, 4] * output[:, 5]  # objectness * class_conf

        for bbox, cls_id, score in zip(bboxes, classes, scores):
            if score < cls_conf:
                continue
            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
            x1_p = max(x1 - padding, 0)
            y1_p = max(y1 - padding, 0)
            x2_p = min(x2 + padding, w)
            y2_p = min(y2 + padding, h)
            if need_frame:
                crop = frame[y1_p:y2_p, x1_p:x2_p]
                crop = np.asarray(crop, dtype=np.uint8)
            else:
                crop = None

            crops.append({
                'crop': crop,
                'img_size': (y2_p - y1_p, x2_p - x1_p),  # (h, w)
                'class': int(cls_id),
                'score': float(score),
                'bbox': (x1_p, y1_p, x2_p, y2_p),
                'track_id': None
            })
    # if shm : shm.close()
    return crops


def _pose_landmarker_process(input_frame_instance_queue, output_frame_instance_queue, model_asset_path, debug=False):
    """
    pose_landmarker를 사용하여 이미지에서 인물(사람)의 랜드마크를 추출하는 워커 프로세스

    Args:
        input_frame_instance_queue (Queue): 입력 프레임 큐 (StreamFrameInstance)
        output_frame_instance_queue (Queue): 포즈 결과 포함된 프레임 출력 큐
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """
    pose_landmarker = PoseDetector(current_process_name=current_process().name, model_asset_path=model_asset_path,
                                   debug=debug)
    try:
        while True:
            stream_frame_instance = input_frame_instance_queue.get()
            if debug: print(f"[DEBUG] instance of pose_landmarker: {stream_frame_instance}")
            if stream_frame_instance is None:
                break
            pose_landmarker_results = pose_landmarker.detect(stream_frame_instance, debug=debug)
            if debug: print(f"[DEBUG] pose_landmarker_results: {pose_landmarker_results}")
            stream_frame_instance.pose_detection_list = pose_landmarker_results
            output_frame_instance_queue.put(stream_frame_instance)
    except KeyboardInterrupt:
        print(f"[DEBUG] instance of pose_landmarker is DIE: {current_process().name}")
    except Exception as e:
        print(f"[ERROR] pose_landmarker process: {e}")


def run_pose_landmarker(input_frame_instance_queue, output_frame_instance_queue, model_asset_path, process_num=8,
                        debug=False):
    """
    포즈 추정 워커 프로세스 다중 실행

    Args:
        input_frame_instance_queue (Queue): 입력 StreamFrameInstance 큐
        output_frame_instance_queue (Queue): 포즈 결과 포함한 출력 큐
        process_num (int): 생성할 워커 프로세스 수
        debug (bool): 디버그 메시지 출력 여부

    Returns:
        list of Process: 실행된 프로세스 객체 리스트
        :param model_asset_path:
    """
    processes = list()
    for i in range(process_num):
        process = Process(name=f"_pose_landmarker_process-{i}", target=_pose_landmarker_process,
                          args=(input_frame_instance_queue, output_frame_instance_queue, model_asset_path, debug))
        process.daemon = True
        process.start()
        if debug: print(f"[DEBUG] pose_landmarker process {i} start")
        processes.append(process)
    return processes


class PoseDetector:
    """
    MediaPipe 기반 포즈 추정기 클래스

    Args:
        current_process_name (str): 프로세스 이름 (디버깅용)
        model_asset_path (str): 모델 파일 경로 (.task 파일)
        num_poses (int): 최대 추정 인원 수
        show_now (bool): 실시간 결과 시각화 여부
        debug (bool): 디버그 출력 여부
    """

    def __init__(self, current_process_name, model_asset_path='pose_landmarker.task', num_poses=4, show_now=False,
                 debug=False):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=num_poses,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.show_now = show_now
        self.debug = debug
        self.current_process_name = current_process_name
        if self.debug: print(f"pose_landmarker: {self.detector}")

    def detect(self, stream_frame_instance, debug=False):
        """
        crop 이미지에서 포즈 추정 실행

        Args:
            stream_frame_instance (StreamFrameInstance): 입력 프레임 객체
            debug (bool): 디버그 출력 여부

        Returns:
            list: 각 객체별 pose 결과 리스트 (MediaPipe Result 또는 빈 리스트)
        """
        crop_object_images = crop_objects(stream_frame_instance)
        pose_landmarker_results = list()

        # 포즈 감지용 화면 초기화
        pose_demo_name = "DRAWN IMAGE of ERROR"
        if self.show_now:
            pose_demo_name = f"DRAWN IMAGE of {self.current_process_name}"
            cv2.namedWindow(pose_demo_name, cv2.WINDOW_NORMAL)

        for crop_object_img in crop_object_images:
            # crop_object_img가 올바른 형식 인지 확인후 변환
            image_data = np.ascontiguousarray(crop_object_img['crop'])
            if image_data.dtype != np.uint8:
                image_data = image_data.astype(np.uint8)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_data
            )
            detection_result = self.detector.detect(mp_image)
            if debug:
                print(f"DETECTED by pose_landmarker: {detection_result}")

            if self.show_now:  # 포즈 감지용 화면 출력
                annotated_image = draw_world_landmarks_with_coordinates(detection_result, crop_object_img['crop'],
                                                                        debug=debug)
                cv2.imshow(pose_demo_name, annotated_image)
                cv2.waitKey(1)

            if detection_result:
                pose_landmarker_results.append(detection_result)
            else:
                pose_landmarker_results.append([])

        if not pose_landmarker_results:
            return None
        return pose_landmarker_results


# noinspection PyUnresolvedReferences
def draw_world_landmarks_with_coordinates(detection_result, rgb_image=None, img_size=None, debug=False):
    """
      포즈 추정 결과를 이미지에 시각화 (랜드마크 + 낙상 텍스트)

      Args:
          detection_result: MediaPipe pose_landmarker 결과 객체
          rgb_image (np.ndarray or None): 시각화할 원본 이미지, None이면 검정 배경 생성
          img_size (tuple or None): rgb_image가 없을 때 사용할 (h, w)
          debug (bool): 디버그 메시지 출력 여부

      Returns:
          np.ndarray: 랜드마크와 낙상 결과가 그려진 이미지

      Raises:
          Exception: 이미지 생성 또는 랜드마크 그리기 중 오류 가능 (내부 미처리)
      """
    # 2D 정규화 랜드 마크 리스트 (list of list)rgb_image
    pixel_landmarks_list = detection_result.pose_landmarks

    # 이미지 미포함 시 검정 화면에 처리
    if rgb_image is None:
        annotated_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    else:
        annotated_image = np.copy(rgb_image)

    # 랜드 마크 없으면 원본 반환
    if not pixel_landmarks_list:
        cv2.putText(annotated_image, "Pose Fail", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 0)
        return annotated_image

    # 감지 결과 추가
    detection_fall_result = fall_detecting_algorithm.detect_fall(detection_result, debug=debug)
    if detection_fall_result is None:
        cv2.putText(annotated_image, "Conf Fail", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 0)
    elif detection_fall_result is True:
        cv2.putText(annotated_image, "FALL", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_image, "NOT FALL", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 인물(사람)별로 순회
    for idx in range(len(pixel_landmarks_list)):
        if detection_fall_result is None:
            continue
        lm2d_list = pixel_landmarks_list[idx]
        if not lm2d_list:
            continue

        # 2D overlay: 연결선 + 점
        pose2d_proto = landmark_pb2.NormalizedLandmarkList()
        pose2d_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in lm2d_list
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose2d_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image

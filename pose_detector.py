from multiprocessing import Process, current_process
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import fall_detecting_algorithm
import numpy as np
import torch
import dataclass_for_StreamFrameInstance
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def crop_objects(stream_frame_instance, padding=10, cls_conf=0.35, need_frame=True, debug=False):
    """
    stream_frame_instance 에서 감지된 객체들을 padding 만큼 여백을 두고 잘라냅니다.
    - padding: 바운딩박스 주변에 추가할 픽셀 여백
    - cls_conf: confidence threshold 이하의 객체는 무시
    Returns: 리스트 of dict, each dict has:
        {
            'crop': np.ndarray,        # 잘라낸 이미지
            'img_size': (h, w),        # 이미지 크기 (h, w)
            'class': int or str,       # 클래스 아이디 (또는 이름)
            'score': float,            # confidence 점수
            'bbox': (x1, y1, x2, y2),  # 잘라낸 영역 (with padding)
            'track_id': int or None    # tracking 경우의 ID (detection 경우 None)
        }
    """
    h, w = stream_frame_instance.height, stream_frame_instance.width
    if need_frame:
        # 1) 원본 프레임 복원
        frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(
            stream_frame_instance, debug=debug
        )
        frame = frame.reshape((h, w, 3))

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
    elif hasattr(stream_frame_instance, 'human_detection_numpy') and stream_frame_instance.human_detection_numpy is not None:
        # detection 된 바운딩박스는 모델 입력 크기(test_size)에 맞춰져 있으므로 원본 크기로 스케일 복원
        test_size = (stream_frame_instance.human_detection_tsize,
                     stream_frame_instance.human_detection_tsize)
        ratio = min(test_size[0] / h, test_size[1] / w)

        output = torch.tensor(stream_frame_instance.human_detection_numpy,
                              dtype=torch.float32)
        if output.numel() == 0:
            return crops

        bboxes = output[:, 0:4] / ratio   # [x1, y1, x2, y2]
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

    return crops

def _pose_landmarker_process(input_frame_instance_queue, output_frame_instance_queue, debug=False):
    """
    pose_landmarker를 사용하여 이미지에서 인물(사람)의 랜드마크를 추출합니다.
    - stream_frame_instance: StreamFrameInstance
    - debug: debug 여부
    Returns: list of dict, each dict has:
        {
            'crop': np.ndarray,        # 잘라낸 이미지
            'class': int or str,       # 클래스 아이디 (또는 이름)
            'score': float,            # confidence 점수
    """
    pose_landmarker = PoseDetector(current_process_name=current_process().name, debug=debug)
    try:
        while True:
            stream_frame_instance = input_frame_instance_queue.get()
            if debug: print(f"[DEBUG] instance of pose_landmarker: {stream_frame_instance}")
            if stream_frame_instance is None:
                break
            pose_landmarker_results = pose_landmarker.detect(stream_frame_instance, debug=debug)
            if debug: print(f"[DEBUG] pose_landmarker_results: {pose_landmarker_results}")
            stream_frame_instance.pose_detection_list=pose_landmarker_results
            output_frame_instance_queue.put(stream_frame_instance)
    except KeyboardInterrupt:
        print(f"[DEBUG] instance of pose_landmarker is DIE: {current_process().name}")
    except Exception as e:
        print(f"[ERROR] pose_landmarker process: {e}")

def run_pose_landmarker(input_frame_instance_queue, output_frame_instance_queue, process_num=8, debug=False):
    processes=list()
    for i in range(process_num):
        process = Process(name= f"_pose_landmarker_process-{i}", target=_pose_landmarker_process, args=(input_frame_instance_queue, output_frame_instance_queue, debug))
        process.daemon = True
        process.start()
        if debug: print(f"[DEBUG] pose_landmarker process {i} start")
        processes.append(process)
    return processes

class PoseDetector:
    def __init__(self, current_process_name, model_asset_path='pose_landmarker.task', num_poses=4, show_now=False, debug=False):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=num_poses,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.show_now=show_now
        self.debug = debug
        self.current_process_name = current_process_name
        if self.debug: print(f"pose_landmarker: {self.detector}")

    def detect(self, stream_frame_instance, debug=False):
        crop_object_images = crop_objects(stream_frame_instance)
        pose_landmarker_results = list()
        crop_object_img=None
        pose_demo_name="DRAWED IMAGE of ERROR"
        if self.show_now:
            pose_demo_name=f"DRAWED IMAGE of {self.current_process_name}"
            cv2.namedWindow(pose_demo_name, cv2.WINDOW_NORMAL)

        for crop_object_img in crop_object_images:
            # 이미지가 올바른 형식인지 확인하고 변환
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

            if self.show_now:
                annotated_image=draw_world_landmarks_with_coordinates(detection_result, crop_object_img['crop'])
                cv2.imshow(pose_demo_name, annotated_image)
                cv2.waitKey(1)

            if detection_result: pose_landmarker_results.append(detection_result)
            else: pose_landmarker_results.append([])

        if not pose_landmarker_results:
            return None
        return pose_landmarker_results


def draw_world_landmarks_with_coordinates(detection_result, rgb_image=None, img_size=None, debug=False):
    # 2D 정규화 랜드마크 리스트 (list of list)rgb_image
    pixel_landmarks_list = detection_result.pose_landmarks
    # 3D 월드 랜드마크 리스트 (list of list)
    world_landmarks_list = detection_result.pose_world_landmarks


    if rgb_image is None:
        #crop_h, crop_w = rgb_image.shape[:2]
        annotated_image=np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    else:
        annotated_image = np.copy(rgb_image)
    #h, w = annotated_image.shape[:2]

    # 둘 중 하나라도 없으면 원본 반환
    if not pixel_landmarks_list or not world_landmarks_list:
        cv2.putText(annotated_image, "Pose Fail", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 0)
        return annotated_image

    #TODO: 분리 요망
    detection_result=fall_detecting_algorithm.detect_fall(detection_result)
    if detection_result is None:
        cv2.putText(annotated_image, "Conf Fail", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 0)
    elif detection_result is True:
        cv2.putText(annotated_image, "FALL", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_image, "NOT FALL", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 인물(사람)별로 순회
    for idx in range(len(pixel_landmarks_list)):
        if detection_result is None:
            continue
        lm2d_list = pixel_landmarks_list[idx]
        lm3d_list = world_landmarks_list[idx]
        if not lm2d_list or not lm3d_list:
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

        # 3D 월드 좌표 텍스트 표시
        #for lm2d, lm3d in zip(lm2d_list, lm3d_list):
        #    x_px = int(lm2d.x * w)
        #    y_px = int(lm2d.y * h)
        #    coord_text = f"({lm3d.x:.2f}m, {lm3d.y:.2f}m, {lm3d.z:.2f}m)"
        #    # 검은 테두리
        #    cv2.putText(annotated_image, coord_text, (x_px + 5, y_px - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
        #    # 흰 글씨
        #    cv2.putText(annotated_image, coord_text, (x_px + 5, y_px - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return annotated_image


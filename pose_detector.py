import time
from multiprocessing import Process, current_process, Queue, Pool
from threading import Thread
from multiprocessing.pool import ThreadPool
from dataclass_for_StreamFrameInstance import StreamFrameInstance
import dataclass_for_StreamFrameInstance
from dataclass_for_StreamFrameInstance import FrameSequentialProcesser
from queue import Empty
import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import partial

import fall_detecting_algorithm


# TODO: mediapipe가 GPU를 사용하도록 할 수 있을 텐데

class PoseDetector:
    """
    MediaPipe 기반 포즈 추정기 클래스

    Args:
        model_asset_path (str): 모델 파일 경로 (.task 파일)
        num_poses (int): 최대 추정 인원 수
        show_now (bool): 실시간 결과 시각화 여부
        debug (bool): 디버그 출력 여부
    """

    def __init__(self, model_asset_path='pose_landmarker.task', num_poses=4,
                 show_now=False, num_process=4,
                 debug=False):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=num_poses,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.show_now = show_now
        self.num_process = num_process
        self.debug = debug

        self.pose_demo_name = "DRAWN IMAGE of ERROR"
        self.instance_crop_image_res_dict = dict() #id:{"instance":INST, "crop_imgs": [IMG], "result":{ing_index:RES} }
        self.orig_instance = dict()


        self.input_q=Queue()
        self.output_q=Queue()

        self.pool_thread= Thread(
            target=pose_detector_pool_process,args=(self.input_q, self.output_q, self.detector, self.num_process, self.debug),
            name = "PosePoolProc"
        )
        self.pool_thread.daemon=True
        self.pool_thread.start()

        if self.debug: print(f"pose_landmarker: {self.detector}")

    def pre_detect(self, stream_frame_instance):
        inst_id=time.perf_counter()

        self.instance_crop_image_res_dict[inst_id] = crop_objects(stream_frame_instance)
        self.orig_instance[inst_id]=stream_frame_instance

        for idx, crop_object_img in enumerate(self.instance_crop_image_res_dict[inst_id]):
            # crop_object_img가 올바른 형식 인지 확인후 변환
            image_data = np.ascontiguousarray(crop_object_img['crop'])
            if image_data.dtype != np.uint8:
                image_data = image_data.astype(np.uint8)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_data
            )
            self.input_q.put({"inst_id":time.perf_counter(), "img_index":idx, "mp_image":mp_image})

    def post_detect(self, stream_frame_instance=None):
        """
        crop 이미지에서 포즈 추정 실행

        Returns:
            list: 각 객체별 pose 결과 리스트 (MediaPipe Result 또는 빈 리스트)
        """

        if stream_frame_instance is not None:
            self.pre_detect(stream_frame_instance)


        while True:
            output_dict = self.output_q.get()
            detection_result = output_dict["detection_result"]
            inst_id=output_dict["inst_id"]
            img_index=output_dict["img_index"]

            self.instance_crop_image_res_dict[inst_id]["result"][img_index]=detection_result

            if self.debug:
                print(f"DETECTED by pose_landmarker: {detection_result}")

            if self.show_now:  # 포즈 감지용 화면 출력
                for crop_object_img in self.instance_crop_image_res_dict:
                    annotated_image = draw_world_landmarks_with_coordinates(detection_result, crop_object_img['crop'],
                                                                            debug=self.debug)
                    cv2.imshow(self.pose_demo_name, annotated_image)
                    cv2.waitKey(1)

            if len(self.instance_crop_image_res_dict[inst_id]["result"][img_index])==len(self.instance_crop_image_res_dict[inst_id]["crop_imgs"]):
                sorted(self.instance_crop_image_res_dict[inst_id]["result"])
                for result_per_img in self.instance_crop_image_res_dict[inst_id]["result"].values():
                    self.orig_instance[inst_id].pose_detection_list.append(result_per_img)
                return self.orig_instance.pop(inst_id)



def pose_detector_pool_worker(mp_image_data_dict_list, detector):
    detection_result = detector.detect(mp_image_data_dict_list.pop("mp_image"))
    mp_image_data_dict_list["detection_result"] = detection_result
    return mp_image_data_dict_list

def pose_detector_pool_process(input_q, output_q, detector, num_process=4, debug=False):
    """
    :param input_q: dict 형태 {"id":ID, "mp_image":mp_image}
    :param output_q:
    :param detector:
    :param num_process:
    :param debug:
    :return:
    """
    pool = Pool(processes=num_process)
    input_data_dict_list=list()
    pool_func = partial(pose_detector_pool_worker, detector=detector)
    while True:
        input_data_dict_list.clear()
        for i in range(num_process):
            try:
                input_data_dict = input_q.get(timeout=1)
                input_data_dict_list.append(input_data_dict)
                if debug: print(f"pose_landmarker pool input: {input_data_dict}")
            except Empty:
                break
        results_dict_list=pool.map(pool_func, input_data_dict_list)
        if debug: print(f"pose_landmarker pool results: {results_dict_list}")
        for result_dict in results_dict_list:
            output_q.put(result_dict)






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





def _pose_landmarker_flow(
    input_q, output_q,
    model_asset_path, process_num: int = 8, debug: bool = False
):
    pose_detector = PoseDetector(model_asset_path=model_asset_path, num_poses=process_num,
                                 num_process=4, debug=debug)
    try:
        while True:
            instance=input_q.get()
            instance.sequence_perf_counter["pose_detector_start"] = time.perf_counter()


            result = pose_detector.post_detect(instance)

            # 3) 결과 다시 큐에
            instance.pose_detection_list=result
            instance.sequence_perf_counter["pose_detector_end"] = time.perf_counter()
            output_q.put(instance)

    except KeyboardInterrupt:
        pass

def run_pose_landmarker_proc(input_frame_instance_queue, output_frame_instance_queue, model_asset_path, process_num=8,
                        debug=False):
    process = Process(
        name="_pose_landmarker_process", target=_pose_landmarker_flow,
        args=(input_frame_instance_queue, output_frame_instance_queue, model_asset_path, process_num, debug)
    )
    process.daemon = False
    process.start()
    if debug: print(f"[DEBUG] pose_landmarker process start")
    return process





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

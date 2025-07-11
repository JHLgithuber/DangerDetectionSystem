#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import multiprocessing
import queue
import time
from multiprocessing import Queue, Process

import numpy as np
import torch
from loguru import logger

import dataclass_for_StreamFrameInstance
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

# TODO: 굳이 YOLOX를 써야 하는가? cv2에 인간 잡아주는것이 있을 텐데

multiprocessing.set_start_method('spawn', force=True)


def gpu_index_generator():
    """
    시스템의 GPU 인덱스를 순환적으로 반환하는 제너레이터.
    예: GPU가 3개면 0 → 1 → 2 → 0 → 1 → 2 → ... 계속 반복
    """
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("GPU가 존재하지 않습니다.")

    idx = 0
    while True:
        print(f"GPU index: {idx}")
        yield idx
        idx = (idx + 1) % gpu_count
    # while True:
    #     yield 1


def _inference_worker(input_queue, output_queue, args, gpu_index, all_object=False, debug_mode=False):
    """
    YOLOX 모델 기반 추론 워커 프로세스 실행
    실질적 초론 메서드 실행
    Args:
        input_queue (Queue): 입력 프레임 큐
        output_queue (Queue): 추론 결과 출력 큐
        args (Namespace): YOLOX 실행 인자
        all_object (bool): 모든 클래스 검출 여부
        debug_mode (bool): 디버그 메시지 출력 여부

    Returns:
        None
    """

    torch.cuda.set_device(gpu_index)
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(f'cuda:{gpu_index}')
    model.eval()

    predictor = Predictor(
        model=model,
        args=args,
        exp=exp,
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy,
        gpu_index=gpu_index,
    )
    if debug_mode: print("predictor init")
    try:
        predictor.inference(input_queue=input_queue, output_queue=output_queue, all_object=all_object,
                            debug_mode=debug_mode)
    except KeyboardInterrupt:
        print("inference_worker_process END")
        return
    except Exception as e:
        print(e)
        print("inference_worker_process KILL by Exception")
        return


class Predictor(object):
    """
    YOLOX 모델 추론기

    Args:
        model: 학습된 YOLOX 모델 객체
        exp: 실험 설정 객체
        cls_names (list): 클래스 이름 리스트
        trt_file (str): TensorRT 모델 파일 (사용 안 함)
        decoder: YOLO 디코더 (선택적)
        device (str): 'cpu' 또는 'gpu'
        fp16 (bool): FP16 추론 여부
        legacy (bool): 구 버전 호환 여부
    """

    def __init__(
            self,
            model,
            args,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
            gpu_index=0,
    ):
        self.gpu_index = gpu_index
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, input_queue, output_queue, all_object=False, debug_mode=False, batch_size=6, max_wait=0.01, ):
        """
        실질적 추론 메서드
        입력 큐로부터 이미지 받아 배치 추론 후 결과 출력 큐에 저장

        Args:
            input_queue (Queue): 추론 대상 프레임 입력 큐
            output_queue (Queue): 추론 결과 전달 큐
            all_object (bool): 모든 클래스 검출 여부
            debug_mode (bool): 디버그 메시지 출력 여부
            batch_size (int): 배치 처리 개수
            max_wait (float): 최대 대기 시간 (초)

        Returns:
            None
        """
        batch_inputs = []
        batch_ids = []
        last_collect = time.time()

        while True:
            input_dict = input_queue.get()

            batch_inputs.append(input_dict["img"])
            batch_ids.append(input_dict["id"])

            if debug_mode: print(f"input_queue get {input_dict['id']}")
            if not (len(batch_inputs) >= batch_size or (batch_inputs and time.time() - last_collect > max_wait)):
                if debug_mode: print("not enough inputs for batch inference")
                continue

            imgs = []
            for img in batch_inputs:
                img, _ = self.preproc(img, None, self.test_size)
                imgs.append(img)
            # (B, C, H, W) 텐서로 변환
            batch_tensor = torch.from_numpy(np.stack(imgs, axis=0)).float()
            if self.device == "gpu":
                batch_tensor = batch_tensor.to(device=f'cuda:{self.gpu_index}')
                if self.fp16:
                    batch_tensor = batch_tensor.half()  # to FP16

            with torch.no_grad():
                t0 = time.time()
                outputs = self.model(batch_tensor)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )

            # 3) 각 프레임별로 후처리 & 큐에 넣기
            infer_time = time.time() - t0
            for out, fid in zip(outputs, batch_ids):
                if out is not None:
                    out = out.cpu().clone()
                    if debug_mode: print(f"output_queue put {fid}")
                    if not all_object:
                        mask = out[:, 6] == 0
                        out = out[mask]
                    output_queue.put({
                        "output_numpy": out,
                        "id": fid,
                        "infer_time": infer_time
                    })
                else:
                    if debug_mode: print(f"output_queue put None {fid}")
                    output_queue.put({
                        "output_numpy": None,
                        "id": fid,
                        "infer_time": infer_time
                    })

            # 4) 배치 초기화
            batch_inputs.clear()
            batch_ids.clear()
            last_collect = time.time()
            # time.sleep(0.0001)


# noinspection PyUnusedLocal
def imageflow_main_proc(args, stream_queue, return_queue, worker_num=4, all_object=False, debug_mode=False, ):
    """
    멀티 추론 워커 구성 및 실시간 추론 흐름 처리
    추론 전후 입출력 처리

    Args:
        args (Namespace): 실행 인자
        stream_queue (Queue): 프레임 입력 큐
        return_queue (Queue): 추론 결과 반환 큐
        worker_num (int): 추론 워커 개수
        all_object (bool): 모든 클래스 검출 여부
        debug_mode (bool): 디버그 출력 여부

    Returns:
        None
    """
    inference_worker_set = set()
    input_queue = Queue(maxsize=32)
    output_queue = Queue(maxsize=32)
    waiting_instance_dict = dict()
    gpu_gen = gpu_index_generator()
    try:
        for index in range(worker_num):
            gpu_index = next(gpu_gen)
            inference_worker_process = Process(name=f"_inference_worker-{index} of GPU{gpu_index}",
                                               target=_inference_worker,
                                               args=(input_queue, output_queue, args, gpu_index, all_object,
                                                     debug_mode))
            inference_worker_process.daemon = True
            inference_worker_process.start()
            if debug_mode: print(f"inference_worker_process of GPU{gpu_index} {inference_worker_process.pid} start")
            inference_worker_set.add(inference_worker_process)

        while True:
            try:
                # 받은 프레임을 추론 워커에 분배
                if not stream_queue.empty():
                    # 큐에서 프레임 객체 꺼내기
                    stream_frame_instance = stream_queue.get()
                    stream_frame_instance.sequence_perf_counter["human_detector_start"]=time.perf_counter()
                    if stream_frame_instance.bypass_flag is False:
                        instance_id = stream_frame_instance.stream_name + '-' + stream_frame_instance.captured_datetime.strftime(
                            "%Y%m%d%H%M%S%f")  # 프레임별 ID생성
                        waiting_instance_dict[instance_id] = stream_frame_instance  # 입력 인스턴스 저장
                        frame = dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance,
                                                                                                debug=debug_mode)
                        input_queue.put({"img": frame, "id": instance_id})
                        if debug_mode: print(f"input_queue put {instance_id}")
                    elif stream_frame_instance.bypass_flag is True:
                        return_queue.put(stream_frame_instance)

                # 추론 완료된 결과를 기존 인스턴스에 매칭 및 삽입 후 리턴
                if not output_queue.empty():
                    output_dict = output_queue.get()
                    if output_dict["id"] in waiting_instance_dict:
                        if return_queue.full():
                            return_queue.get()
                        output_frame_instance = waiting_instance_dict.pop(output_dict["id"])  # id로 인스턴스 매칭
                        if output_dict["output_numpy"] is None:
                            output_frame_instance.human_detection_numpy = None
                        else:
                            output_frame_instance.human_detection_numpy = output_dict["output_numpy"]  # 추론결과 삽입
                        output_frame_instance.sequence_perf_counter["human_detector_end"] = time.perf_counter()
                        return_queue.put(output_frame_instance)
                    else:
                        logger.info("output_dict id not found")


            except queue.Empty:
                continue

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        for inference_worker_process in inference_worker_set:
            inference_worker_process.terminate()
            inference_worker_process.join()
        logger.info("inference_worker_process terminated")
        logger.info("input_queue closed")


def main(exp, args, stream_queue, return_queue, process_num=4, all_object=False, debug_mode=False):
    """
    YOLOX 추론 파이프라인 전체 초기화 및 메인 프로세스 실행

    Args:
        exp: 실험 설정 객체 (YOLOX Experiment)
        args (Namespace): 실행 인자
        stream_queue (Queue): 실시간 프레임 입력 큐
        return_queue (Queue): 추론 결과 출력 큐
        process_num (int): 추론 워커 프로세스 개수
        all_object (bool): 모든 클래스 검출 여부
        debug_mode (bool): 디버그 출력 여부

    Returns:
        Process: 실행된 메인 프로세스 객체
    """
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    imageflow_demo_process = Process(
        name="imageflow_demo_MAIN_process",
        target=imageflow_main_proc,
        args=(args, stream_queue, return_queue, process_num, all_object, debug_mode)
    )
    imageflow_demo_process.daemon = False
    return imageflow_demo_process

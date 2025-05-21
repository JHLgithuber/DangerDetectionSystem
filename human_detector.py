#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import multiprocessing
import uuid
import argparse
import os
import time
from loguru import logger
import cv2
import torch
import numpy as np

import dataclass_for_StreamFrameInstance
import demo_viewer
import queue
from multiprocessing import Queue, Process, Manager, shared_memory
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import stream_input
from collections import defaultdict

multiprocessing.set_start_method('spawn', force=True)


def _inference_worker(input_queue, output_queue, args, all_object=False, debug_mode=False):
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    predictor = Predictor(
        model=model,
        exp=exp,
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy
    )
    if debug_mode: print("predictor init")
    try: predictor.inference(input_queue=input_queue, output_queue=output_queue, all_object=all_object, debug_mode=debug_mode)
    except KeyboardInterrupt:
        print("inference_worker_process END")
        return
    except Exception as e:
        print(e)
        print("inference_worker_process KILL by Exception")
        return

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
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
        # 실질적 추론 메서드
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
                batch_tensor = batch_tensor.cuda()  # GPU로 업로드
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
                        "output_numpy": out.numpy(),
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
            #time.sleep(0.0001)


def imageflow_demo(predictor, args, stream_queue, return_queue, worker_num=4, all_object=False, debug_mode=False,):
    inference_worker_set = set()
    input_queue = Queue(maxsize=32)
    output_queue = Queue(maxsize=32)
    waiting_instance_dict = dict()

    try:
        for _ in range(worker_num):
            inference_worker_process = Process(target=_inference_worker, args=(input_queue, output_queue, args, all_object, debug_mode))
            inference_worker_process.daemon = True
            inference_worker_process.start()
            if debug_mode: print(f"inference_worker_process {inference_worker_process.pid} start")
            inference_worker_set.add(inference_worker_process)
    
        while True:
            try:
                if not stream_queue.empty():
                    # 큐에서 프레임 객체 꺼내기
                    stream_frame_instance = stream_queue.get()
                    if stream_frame_instance.bypass_flag is False:
                        instance_id = stream_frame_instance.stream_name +'-'+ stream_frame_instance.captured_datetime.strftime(
                            "%Y%m%d%H%M%S%f")
                        waiting_instance_dict[instance_id] = stream_frame_instance
                        frame=dataclass_for_StreamFrameInstance.load_frame_from_shared_memory(stream_frame_instance, debug=True)
                        input_queue.put({"img": frame, "id": instance_id})
                        if debug_mode: print(f"input_queue put {instance_id}")
                    elif stream_frame_instance.bypass_flag is True:
                        return_queue.put(stream_frame_instance)


                if not output_queue.empty():
                    output_dict = output_queue.get()
                    if output_dict["id"] in waiting_instance_dict:
                        if return_queue.full():
                            return_queue.get()
                        output_frame_instance = waiting_instance_dict.pop(output_dict["id"])
                        output_frame_instance.human_detection_numpy = output_dict["output_numpy"]
                        return_queue.put(output_frame_instance)
                    else:
                        logger.info("output_dict id not found")
    
    
            except queue.Empty:
                continue
    
    finally:
        for inference_worker_process in inference_worker_set:
            inference_worker_process.terminate()
            inference_worker_process.join()
        logger.info("inference_worker_process terminated")
        logger.info("input_queue closed")
    


def main(exp, args, stream_queue, return_queue, process_num=4, all_object=False, debug_mode=False):
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

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        args.device, args.fp16, args.legacy,
    )

    imageflow_demo_process = Process(
        target=imageflow_demo,
        args=(predictor, args, stream_queue, return_queue, process_num, all_object, debug_mode)
    )
    imageflow_demo_process.daemon = False
    return imageflow_demo_process


def get_args():
    hard_args = argparse.Namespace(
        demo="video",
        experiment_name=None,
        name="yolox-s",
        path="streetTestVideo.mp4",
        camid=0,
        show_result=True,
        exp_file=None,
        ckpt="yolox_s.pth",
        device="gpu",
        conf=0.45,  #신뢰도
        nms=0.65,   #클수록 겹치는 바운딩박스 제거
        tsize=640,
        fp16=False,
        legacy=False,
        fuse=False,
        trt=False
    )
    return hard_args


if __name__ == "__main__":
    args = get_args()
    exp = get_exp(args.exp_file, args.name)

    debugMode = True
    # showMode=True
    stream_queue = Manager().Queue(maxsize=128)
    return_queue = Manager().Queue(maxsize=128)

    demo_viewer.start_imshow_demo(stream_queue=return_queue)
    time.sleep(1)
    detector_process = main(exp, args, stream_queue, return_queue)
    detector_process.start()
    time.sleep(3)
    testStreamList = [
        stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=stream_queue,
                                stream_name="TEST_0", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_1", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_2", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_3", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_4", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_5", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_6", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_7", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_8", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", manager_queue=stream_queue,
        #                        stream_name="TEST_9", debug=debugMode),
        ]
    detector_process.join()
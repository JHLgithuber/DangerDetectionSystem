#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import uuid
import argparse
import os
import time
from loguru import logger
import cv2
import torch
import numpy as np
import demo_viewer
import queue
from multiprocessing import Queue, Process, Manager
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import stream_input

def _inference_worker(input_queue, output_queue, args, all_object=False):
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
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

    predictor.inference(input_queue=input_queue, output_queue=output_queue, all_object=all_object)

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

    def inference(self, input_queue, output_queue, all_object=False):
        #실질적 추론 메서드
        while(True):
            if input_queue.empty():
                time.sleep(0.001)
                continue
            input_dict=input_queue.get()
            img = input_dict["img"]
            id = input_dict["id"]

            img, _ = self.preproc(img, None, self.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.float()
            if self.device == "gpu":
                img = img.cuda() #GPU로 업로드
                if self.fp16:
                    img = img.half()  # to FP16

            with torch.no_grad():
                t0 = time.time()
                outputs = self.model(img)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )

            output = outputs[0]
            if output is not None:
                output = output.cpu().clone()
                # 사람 필터링도 여기서
                if not all_object:
                    mask = output[:, 6] == 0
                    output = output[mask]
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
                output_queue.put({"output_numpy":output.numpy(),"id":id})
            else:
                logger.info("Infer FAIL time: {:.4f}s".format(time.time() - t0))
                output_queue.put({"output_numpy":None,"id":id})



def imageflow_demo(predictor, args, stream_queue, return_queue, worker_num=4, all_object=False):
    infrence_worker_set=set()
    input_queue=Queue(maxsize=128)
    output_queue=Queue(maxsize=128)
    waiting_instance_dict=dict()
    for _ in range(worker_num):
        infrence_worker_process=Process(target=_inference_worker,args=(input_queue,output_queue,args,all_object))
        infrence_worker_process.daemon=True
        infrence_worker_process.start()
        infrence_worker_set.add(infrence_worker_process)

    while True:
        try:
            if not stream_queue.empty():
                # 큐에서 프레임 객체 꺼내기
                stream_frame_instance = stream_queue.get(timeout=0.001)
                instance_id=stream_frame_instance.stream_name + stream_frame_instance.captured_datetime.strftime("%Y%m%d%H%M%S%f")
                waiting_instance_dict[instance_id]=stream_frame_instance

                # 바이트 데이터를 NumPy 배열로 변환 (OpenCV 형식으로 복원)
                frame = np.frombuffer(stream_frame_instance.row_frame_bytes, dtype=np.uint8)
                frame = frame.reshape(stream_frame_instance.height, stream_frame_instance.width, 3)

                # 추론 수행
                input_queue.put({"img":frame,"id":instance_id})

            if not output_queue.empty():
                output_dict=output_queue.get()
                if output_dict["id"] in waiting_instance_dict:
                    if return_queue.full():
                        return_queue.get()
                    output_frame_instance=waiting_instance_dict.pop(output_dict["id"])
                    output_frame_instance.human_detection_numpy=output_dict["output_numpy"]
                    return_queue.put(output_frame_instance)
                else:
                    logger.info("output_dict id not found")


        except queue.Empty:
            continue


def main(exp, args, stream_queue, return_queue):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

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
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        args.device, args.fp16, args.legacy,
    )

    imageflow_demo_process = Process(
        target=imageflow_demo,
        args=(predictor, args, stream_queue, return_queue, 4, True)
    )
    imageflow_demo_process.daemon=False
    return imageflow_demo_process

def get_args():
    hard_args = argparse.Namespace(
        demo="video",
        experiment_name=None,
        name="yolox-x",
        path="streetTestVideo.mp4",
        camid=0,
        show_result=True,
        exp_file=None,
        ckpt="yolox_x.pth",
        device="gpu",
        conf=0.25,
        nms=0.45,
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

    debugMode=True
    #showMode=True
    stream_queue = Manager().Queue(maxsize=128)
    return_queue = Manager().Queue(maxsize=128)


    demo_viewer.start_imshow_demo(stream_queue=return_queue)
    time.sleep(1)
    detector_process=main(exp, args, stream_queue, return_queue)
    detector_process.start()
    time.sleep(3)
    testStreamList= [#stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=stream_queue, stream_name="TEST_0", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv069.stream", manager_queue=stream_queue, stream_name="TEST_1", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv070.stream", manager_queue=stream_queue, stream_name="TEST_2", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv071.stream", manager_queue=stream_queue, stream_name="TEST_3", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv072.stream", manager_queue=stream_queue, stream_name="TEST_4", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv073.stream", manager_queue=stream_queue, stream_name="TEST_5", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv074.stream", manager_queue=stream_queue, stream_name="TEST_6", debug=debugMode),
        #stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv075.stream", manager_queue=stream_queue, stream_name="TEST_7", debug=debugMode),
        stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv076.stream", manager_queue=stream_queue, stream_name="TEST_8", debug=debugMode),
        stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv077.stream", manager_queue=stream_queue, stream_name="TEST_9", debug=debugMode), ]
    detector_process.join()

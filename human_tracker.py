#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import cv2
import torch
import numpy as np
from multiprocessing import Queue, Process, Manager
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import stream_input

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

    def inference(self, img):
        #실질적 추론 메서드
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

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

        # 사람만 필터링 (클래스 0)
        if outputs[0] is not None:
            mask = outputs[0][:, 6] == 0
            outputs[0] = outputs[0][mask]

        output = outputs[0].cpu()   #CPU로 다운로드

        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return output, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img.copy(), bboxes, scores, cls, cls_conf, self.cls_names)   #프레임에 결과 그려줌
        return vis_res


def imageflow_demo(predictor, args, stream_queue):
    while True:
        if not stream_queue.empty():
            # 큐에서 프레임 객체 꺼내기
            stream_frame_instance = stream_queue.get()

            # 바이트 데이터를 NumPy 배열로 변환 (OpenCV 형식으로 복원)
            frame = np.frombuffer(stream_frame_instance.row_frame_bytes, dtype=np.uint8)
            frame = frame.reshape(stream_frame_instance.height, stream_frame_instance.width, 3)

            # 추론 수행
            output, img_info = predictor.inference(frame)  # TODO: 사전에 올려놓을 수 있게 해야


            # 결과 시각화 (박스 그리기)
            result_frame = predictor.visual(output, img_info, predictor.confthre)

            if args.save_result:
                # 결과를 파일로 저장하거나, GUI 화면에 출력
                # vid_writer.write(result_frame)  # 파일로 저장 시 사용

                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)  # GUI화면으로 출력
                cv2.imshow("yolox", result_frame)

                # ESC 또는 Q 키 입력 시 종료
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
        else:
            # 큐가 비었으면 약간 대기 (CPU 과점유 방지)
            time.sleep(0.001)


def main(exp, args, stream_queue):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.trt:
        args.device = "gpu"

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

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
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
    imageflow_demo(predictor=predictor, args=args, stream_queue=stream_queue)

def get_args():
    hard_args = argparse.Namespace(
        demo="video",
        experiment_name=None,
        name="yolox-x",
        path="streetTestVideo.mp4",
        camid=0,
        save_result=True,
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
    manager = Manager()

    test_stream=stream_input.RtspStream(rtsp_url="rtsp://210.99.70.120:1935/live/cctv068.stream", manager_queue=manager.Queue(), stream_name="TEST_0", debug=debugMode)
    stream_queue=test_stream.get_stream_queue()

    main(exp, args, stream_queue)

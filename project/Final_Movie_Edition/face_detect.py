import sys

sys.path.append("../")
import numpy as np
import math
import pickle
import cv2
from PIL import Image, ImageDraw, ImageFont

import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def detect(opt, img0):
    crop_image_list = []
    xyxy_list = []
    imgsz = 640
    weights = r'model/last.pt'

    # Initialize
    device = torch_utils.select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Padded resize
    img = letterbox(img0)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    dataset = [['', img, img0, '']]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                    image = np.array(image)
                    crop_img = image[y1:y2, x1:x2]
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                    crop_image_list.append(crop_img)
                    xyxy_list.append([x1, y1, x2, y2])
    return crop_image_list, xyxy_list


def setOPT():
    # 文件配置
    # *******************************************************

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:/py/FaceRecognition/weights/last.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='C:/Users/lieweiai/Desktop/26321934-1-192.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='../inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--facenet-model-path', type=str, default='D:/code_data/facenet/20180402-114759',
                        help='miss facenet-model')
    parser.add_argument('--svc-path', type=str, default='D:/code_data/face_recognition/pkl/SVCmodel.pkl',
                        help='miss svc')
    parser.add_argument('--database-path', type=str, default='D:/code_data/face_recognition/npz/Database.npz',
                        help='miss database')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)

    return opt

    # *******************************************************


def face_detect(img):
    return detect(setOPT(), img)


if __name__ == "__main__":
    image_address = r'inference/images/2.jpg'
    img = cv2.imread(image_address)
    image_list, xyxy_list = face_detect(img)
    print(xyxy_list)
    for face_image in image_list:
        cv2.namedWindow("Faces")
        cv2.imshow("Faces", face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

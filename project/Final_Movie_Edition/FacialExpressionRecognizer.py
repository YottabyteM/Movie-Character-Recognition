# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : FacialExpressionRecognizer.py
# Time       ：2022/7/20 16:26
# Author     ：沈冠翔
# version    ：python 3.8.10
# Description：
"""

import numpy as np
import os
from tensorflow import keras
import cv2
import Fer_Config
from face_detect import face_detect

# 去除红色的提示信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FacialExpressionRecognizer:

    def __init__(self, model_path=Fer_Config.model_path):
        """
        加载训练好的表情识别模型
        """
        print("从%s加载表情识别模型..." % (model_path,))
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.model = keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)
            self.input_img_shape = (Fer_Config.height, Fer_Config.width)  # 获取CNN输入层的row、col
            print("表情模型概要：")
            self.model.summary()
            # keras.utils.plot_model(self.model,
            #                        to_file='FER_CNN.png',
            #                        show_shapes=True,
            #                        show_dtype=True,
            #                        show_layer_names=True)
        else:
            print("模型路径出错")

    def predict_face_img(self, test_img):
        """
        :param test_img: 单张人脸图片
        :return: 表情识别网络分类得到的前两个标签，以及标签的置信概率
        """
        resized_img = cv2.resize(test_img, self.input_img_shape, interpolation=cv2.INTER_AREA)  # 首先裁剪/扩充原图
        x = np.expand_dims(np.array(resized_img) / 255.0, axis=0)  # 转化为ndarray、标准化、添加batch维度，以便作为CNN的输入
        pred_x = self.model.predict(x, verbose=1)[0]
        pred_arg_x = np.argsort(pred_x, axis=0)
        class0 = pred_arg_x[-1]  # 最大概率的表情类别
        class1 = pred_arg_x[-2]  # 概率第二大的表情类别
        return Fer_Config.class_names[class0], pred_x[class0], Fer_Config.class_names[class1], pred_x[class1]

    def recognize_img(self, img, faces, show=True, mark_confidence=True):
        """
        :param img: 单张图片
        :param faces: 图片中所有人脸框的位置(x1,y1,x2,y2)
        :param show: 是否在程序中显示新图片
        :param mark_confidence: 是否标注置信概率
        :return: 将该图片标注好表情后得到的新图片
        将单张图片中的人脸表情信息标注出来
        """
        print("正在识别并标注表情...")
        for x1, y1, x2, y2 in faces:
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label0, confidence0, label1, confidence1 = self.predict_face_img(img[y1:y2, x1:x2])
            text = label0 + '(' + str(round(confidence0 * 100, 2)) + '%)' if mark_confidence else label0
            if mark_confidence and confidence0 < 0.5:
                text += ' ' + label1 + '(' + str(round(confidence1 * 100, 2)) + '%)'
            cv2.putText(img, text=text, org=(x1, y2 + 23), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5,
                        color=(255, 0, 0), thickness=2)
        if show:
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    def recognize_video(self, video_in_path, video_out_path='processed_video.mp4'):
        """
        :param video_in_path: 原始视频路径
        :param video_out_path: 处理后的视频的保存路径
        通过逐帧处理的方法将视频中的人脸表情信息标注出来
        """
        capture = cv2.VideoCapture(video_in_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)
        if capture.isOpened():
            while True:
                ret, img = capture.read()
                if not ret:
                    break
                _, bboxes = face_detect(img)
                img = self.recognize_img(img, bboxes, show=False, mark_confidence=False)
                writer.write(img)
        else:
            print('视频打开失败！')
        capture.release()
        writer.release()


if __name__ == '__main__':
    recognizer = FacialExpressionRecognizer()
    img1 = cv2.imread("photo/li-yi-feng/20220107210635_b5471.jpeg", flags=cv2.IMREAD_COLOR)
    _, faces1 = face_detect(img1)
    recognizer.recognize_img(img1, faces1)

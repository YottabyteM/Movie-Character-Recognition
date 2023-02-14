from Pose_detect import Pose_detect
from Get_photo import catcher
from Gender_Age import Gender_Age
from face_recognition import face_reg
from face_detect import face_detect
from FacialExpressionRecognizer import FacialExpressionRecognizer
import numpy as np
import cv2
import argparse


class main_model():
    # 每个人把自己需要的参数写入init中，并且在main.py中加入进行修改
    def __init__(self, name, Pose_detectprotoFile, Pose_detectweightsFile, Pose_detectargs, ActorPath):
        # 初始化参数
        self.name = name
        # 加载动作识别模型
        self.Pose_detectModel = Pose_detect(Pose_detectprotoFile, Pose_detectweightsFile, Pose_detectargs)
        self.args = Pose_detectargs
        # 加载性别预测模型
        self.Gender_AgeModel = Gender_Age()

        self.FaceModel = face_reg(actorpath=ActorPath)
        # 加载人脸识别模型

        # 加载表情识别模型
        self.FacialExpressionRecognizer = FacialExpressionRecognizer()

    def judge(self):
        if self.args == "cpu":
            self.Pose_detectModel.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif self.args == "gpu":
            self.Pose_detectModel.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.Pose_detectModel.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")

    def predict(self, VedioPath):
        # 获取明星的照片
        self.judge()
        print("获取照片中")
        for i in self.name:
            if i != "":
                photo_catcher = catcher(i)
                photo_catcher.run()

        self.FaceModel.train_face()
        # 进行面部识别训练

        ## 进行视频处理
        print("处理中")
        # 读取视频
        vs = cv2.VideoCapture(VedioPath)
        writer = None
        # 循环处理
        while True:
            # 读取每一帧
            (grabbed, frame) = vs.read()

            # 判断视频是否结束
            if not grabbed:
                print("无视频读取...")
                break

            frameClone = frame.copy()
            # Yolo依据frame在frameClone上标记
            # frameClone, bboxes = self.Gender_AgeModel.getFaceBox(self.Gender_AgeModel.faceNet, frame) 这一句话用于跑通我自己的模型
            _, bboxes = face_detect(frame)
            # 返回一个box
            # 之后戴廷钧面部识别、沈冠翔表情识别、马琦性别及年龄预测就依照frameClone做标记
            #################################################################
            ######################                      #####################
            ######################                      #####################
            # 人脸识别
            frameClone = self.FaceModel.predict_face(frameClone)
            # 表情识别
            frameClone = self.FacialExpressionRecognizer.recognize_img(frameClone, bboxes, show=False,
                                                                       mark_confidence=True)
            # 性别预测
            frameClone = self.Gender_AgeModel.predict(bboxes, frame, frameClone)
            ######################                      #####################
            #################################################################

            # 动作识别
            frameClone = self.Pose_detectModel.Predict(frame, frameClone)

            frameFinal = frameClone

            # 把图片写入到视频
            if writer is None:
                # 初始化视频写入器
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    "videos\\result.mp4",
                    fourcc, 30, (frameClone.shape[1], frameClone.shape[0]), True)
            writer.write(frameFinal)

        print("结束...")
        writer.release()
        vs.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run keypoint detection')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    args = parser.parse_args()
    test_model = main_model("李易峰", "./model/coco/pose_deploy_linevec.prototxt",
                            "./model/coco/pose_iter_440000.caffemodel", args, ".\photo")
    test_model.predict("videos/3.mp4")

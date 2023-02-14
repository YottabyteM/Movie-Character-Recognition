import cv2 as cv
import time


class Gender_Age():
    def __init__(self):
        self.faceProto = "./model/Gender_Age/opencv_face_detector.pbtxt"
        self.faceModel = "./model/Gender_Age/opencv_face_detector_uint8.pb"
        self.ageProto = "./model/Gender_Age/age_deploy.prototxt"
        self.ageModel = "./model/Gender_Age/age_net.caffemodel"
        self.genderProtol = "./model/Gender_Age/gender_deploy.prototxt"
        self.genderModel = "./model/Gender_Age/gender_net.caffemodel"
        # 人脸检测的网络和模型
        self.ageNet = cv.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv.dnn.readNet(self.genderModel, self.genderProtol)
        self.faceNet = cv.dnn.readNet(self.faceModel, self.faceProto)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']
    # 检测人脸并绘制人脸bounding box
    def getFaceBox(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
        frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()  # 网络进行前向传播，检测人脸
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                             8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
        return frameOpencvDnn, bboxes

    def predict(self, bboxes, frame, frameClone):
        padding = 20

        if not bboxes:
            print("No face Detected, Checking next frame")
            return None

        for bbox in bboxes:
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                   max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
            #
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)   # blob输入网络进行性别的检测
            genderPreds = self.genderNet.forward()   # 性别检测进行前向传播
            gender = self.genderList[genderPreds[0].argmax()]   # 分类  返回性别类型

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]

            label = "{},{}".format(gender, age)
            cv.putText(frameClone, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                       cv.LINE_AA)
            return frameClone
import cv2
import os
from PIL import Image
import numpy as np
from face_detect import face_detect


class face_reg:
    def __init__(self, actorpath, trainpath=r'trainer.yml'):
        self.actorpath = actorpath
        # 演员图片路径
        self.trainpath = trainpath
        # 训练权重文件地址
        self.ids = []
        # 每个演员对应有多少张照片
        self.actor_names = []

    def record_face(self, img):
        # 输入待标记的图片
        list_img, _ = face_detect(img)
        if len(list_img) != 1:
            print("不符合要求")
            return None
        else:
            gray_img = cv2.cvtColor(list_img[0], cv2.COLOR_BGR2GRAY)
            return gray_img

    def train_face(self):
        faces = []
        # 储存人脸数据(该数据为二位数组)
        actorpaths = [os.path.join(self.actorpath, f) for f in os.listdir(self.actorpath)]
        # 有多少个明星
        for actorpath in actorpaths:
            imagepaths = [os.path.join(actorpath, f) for f in os.listdir(actorpath)]
            self.ids.append(len(imagepaths))
            actor_name = os.path.split(actorpath)[1]
            self.actor_names.append(actor_name)
            for imagePath in imagepaths:  # 遍历列表中的图片
                img = cv2.imread(imagePath)
                gray_img = self.record_face(img)
                if gray_img is None:
                    self.ids[-1] -= 1
                    continue
                else:
                    img_numpy = np.array(gray_img, 'uint8')  # 将图像转化为数组
                    faces.append(img_numpy)
        num = sum(self.ids)
        id_array = [[_i] for _i in range(num)]
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # 获取训练对象
        recognizer.train(faces, np.array(id_array))
        recognizer.write(self.trainpath)  # 保存生成的人脸特征数据文件


    def predict_face(self, img):
        recogizer = cv2.face.LBPHFaceRecognizer_create()
        recogizer.read(self.trainpath)  # 获取脸部特征数据文件

        list_face, list_xyxy = face_detect(img)
        mix = zip(list_face, list_xyxy)
        for (face, xyxy) in mix:
            gray_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # 将检测为人脸的部分画框
            ids, confidence = recogizer.predict(gray_img)  # 进行预测、评分
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            if confidence > 80:
                cv2.putText(img, str("unknown"), (xyxy[0] + 10, xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 1)
            else:
                cv2.putText(img, str(self.set_name(ids)), (xyxy[0] + 10, xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 1)
                # 把姓名打到人脸的框图上

        return img

    def set_name(self, i):
        count = 0
        for id, name in self.ids, self.actor_names:
            count += id
            if count <= i:
                return name


if __name__ == '__main__':
    path = r'inference/images/2.jpg'
    cv2.imread(path)


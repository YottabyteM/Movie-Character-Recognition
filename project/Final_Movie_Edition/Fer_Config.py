# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：2022/7/20 14:48
# Author     ：沈冠翔
# version    ：python 3.8.10
# Description：相关配置
"""

# 模型路径，为相对路径
model_path = 'model/Expression/FER_model.h5'

# CNN输入形状

height = 48
width = 48

# CNN输出的分类结果

class_names = ['Angry', 'Disgusted', 'Feared', 'Happy', 'Sad', 'Surprised', 'Neutral', 'Contemptuous']
num_classes = len(class_names)


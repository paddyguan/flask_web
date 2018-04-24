# -*- coding=utf-8 -*-
# author: paddyguan
# date: 2018/3/18
# 图片预处理
import numpy as np
import os
import cv2
import re
from PIL import Image


# 得到一共多少个样本
def getnum(file_path):
    pathDir = os.listdir(file_path)
    i = 0
    for allDir in pathDir:
        i += 1
    return i


# 制作数据集
def data_label(path, count):
    data = np.empty((count, 1, 128, 192), dtype='float32')  # 建立空的四维张量类型32位浮点
    label = np.empty((count,), dtype='uint8')
    i = 0
    pathDir = os.listdir(path)
    for each_image in pathDir:
        all_path = os.path.join('%s%s' % (path, each_image))  # 路径进行连接
        image = cv2.imread(all_path, 0)
        mul_num = re.findall(r"\d", all_path)  # 寻找字符串中的数字，由于图像命名为300.jpg 标签设置为0
        num = int(mul_num[0]) - 3
        #        print num,each_image
        #        cv2.imshow("fad",image)
        #        print child
        array = np.asarray(image, dtype='float32')
        array -= np.min(array)
        array /= np.max(array)
        data[i, :, :, :] = array
        label[i] = int(num)
        i += 1
    return data, label


# 修改图片尺寸
def ResizeImg(img_in_file, img_out_file):
    w, h = 120, 120
    type = 'png'
    ori_img = Image.open(img_in_file)
    out = ori_img.resize((w, h), Image.ANTIALIAS)
    out.save(img_out_file, type)


if __name__ == "__main__":
    print('pic_preprocess')

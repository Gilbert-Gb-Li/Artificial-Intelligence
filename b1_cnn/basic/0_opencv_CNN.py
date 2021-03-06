#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
open cv 用法及卷积扩展
---------------------
    cv2读入的数据为numpy格式
"""
print(__doc__)

import cv2
import numpy as np
''' ##
接收视频数据
参数：
    0，为第一个摄像头；1为第二个...
    也可以接收路径，导入视频
## '''
cap = cv2.VideoCapture(0)
'''----------------------
平均模糊: kernel值全为1且除以16取均值
    对应CNN中的效果为平均池化
高斯模糊: 与平均模糊类似，
    只不过kernel服从高斯分布
--------------------------'''
# kernel = np.ones([4, 4]) / 16
'''-----------------------
哈尔小波:
    行与列方向一半为1，一半为-1
    由此仅在黑白边界部分有值，其他部分均为0
    实现提取边界的效果
    对应CNN最大池化
-------------------------'''
# kernel = np.ones([4, 4])
'''-----------------
# 提取横向边界
# kernel[:2, :] = -1
# 提取纵向边界
# kernel[:, 2:] = -1
--------------------'''
# kernel[:2, :2] = -1
# kernel[2:, 2:] = -1

while True:
    ''' ##
    ret     : 接收的键盘信号
    frame   : 每一帧的信息
    ## '''
    ret, frame = cap.read()
    ''' ##
    参数	    |   描述
    src	        |   原图像
    dst	        |   目标图像，与原图像尺寸和通过数相同
    ddepth	    |   目标图像的所需深度
    kernel	    |   卷积核（或相当于相关核），单通道浮点矩阵;
                    如果要将不同的内核应用于不同的通道，请使用拆分将图像拆分为单独的颜色平面，然后单独处理它们。
    anchor	    |   内核的锚点，指示内核中过滤点的相对位置;锚应位于内核中;默认值（-1，-1）表示锚位于内核中心。
    detal	    |   在将它们存储在dst中之前，将可选值添加到已过滤的像素中。类似于偏置。
    borderType  |	像素外推法，参见BorderTypes
    ## '''
    # frame = cv2.filter2D(frame, -1, kernel)
    ''' ##
    图像展示
    参数： 图像名字，帧
    ## '''
    cv2.imshow("name", frame)
    ''' ##
    每隔多少毫秒读取一张图像
        1秒有30帧的算法 1000 // 30, // 为下取整除法
        0, 为阻塞
    ## '''
    key = cv2.waitKey(1000//30)


'''------------------------------'''


''' ##
图像读取
    需要waitKey(0)阻塞
    格式 unit8
## '''
img = cv2.imread('data/face.jpg')
'''# 通道格式: blue, green, red'''
cv2.imshow('face', img)
cv2.waitKey(0)
''' ##
 更改通道格式为 rgb，等同于以上方法
## '''
plt.imshow(img[:, :, ::-1])
plt.show()

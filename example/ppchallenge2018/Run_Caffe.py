#coding:utf-8

import re
import time
import numpy
import sys
import os
os.environ['GLOG_minloglevel'] = '2' # suppress Caffe verbose prints
import caffe
import requests
import json

import logging
logger = logging.getLogger("user_activity")

from Validation_BasicFunction import mode, Test_Set, readYUVFile, calculatePSNR 


# 进行caffe模型的深度学习
def net_forward(net, image, patch_size):

    # 存储图像的像素信息
    res = image.copy()

    # 每隔patch_size对图像进行分割计算
    for sy in range(0, image.shape[0], patch_size):
        for sx in range(0, image.shape[1], patch_size):

            # 区块的另一端坐标。（sy，sx）为一端坐标
            ey = min(image.shape[0], sy+patch_size)
            ex = min(image.shape[1], sx+patch_size)

            # 对像素值进行归一化
            patch = image[sy:ey, sx:ex] / 255.0

            # 将归一化的像素值存入到net模型中
            net.blobs['data'].reshape(1, 1, patch.shape[0], patch.shape[1])
            net.blobs['data'].data[...] = patch

            # 深度学习运行（前向网络）
            output = net.forward()

            # 转为无符号整数（0 到 255）
            res_patch = numpy.uint8(numpy.round_(output['res'][0][0] * 255.0))
            res[sy:ey, sx:ex] = res_patch
    return res



# 对图片进行评分的核心程序	
def caffe_validation(prototxt_file, caffeModels, patchSize):

    psnr_accu = 0.0
    counter = 0
    t = time.time()

    # 启动GPU
    caffe.set_device(0)
    caffe.set_mode_gpu()
	
    qp_list = [38, 45, 52]

    for ii in range(3):

        # 参数
        qp = qp_list[ii]

        # caffe模型文件的路径
        prototxt_path = prototxt_file
        caffeModels_path = caffeModels[ii]

        # 模型结构、使用测试模式、模型训练权重(训练中不能执行dropout)        
        net = caffe.Net(prototxt_path, caffe.TEST, weights = caffeModels_path)

        for f in os.listdir('./%s/%s_Q%d_yuv' % (Test_Set, mode, qp)):

            # 获取图片的序号、宽、高
            tmpl = re.split('\_|x|\.', f)
            i = int(tmpl[0])
            w = int(tmpl[1])
            h = int(tmpl[2])

            # 读取测试集的YUV文件
            (Yo, Uo, Vo) = readYUVFile('./%s/%s_yuv/%d.yuv' % (Test_Set, mode, i), w, h)
            (Yd, Ud, Vd) = readYUVFile('./%s/%s_Q%d_yuv/%s' % (Test_Set, mode, qp, f), w, h)

            # 在3个参数下对图像进行深度学习
            Yr = net_forward(net, Yd, patchSize)
            Ur = net_forward(net, Ud, patchSize)
            Vr = net_forward(net, Vd, patchSize)

            # 计算PSNR
            a_psnr_y = calculatePSNR(Yo, Yr)
            a_psnr_u = calculatePSNR(Uo, Ur)
            a_psnr_v = calculatePSNR(Vo, Vr)
            psnr_aft = (6.0 * a_psnr_y + a_psnr_u + a_psnr_v) / 8.0
            psnr_accu += psnr_aft

            # 计数
            counter += 1
            print ('counter = %3d : psnr_aft = %f , total cost time = %f' % (counter, psnr_aft, time.time()-t))

    return psnr_accu / counter, time.time()-t





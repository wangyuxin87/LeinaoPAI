#coding:utf-8

import sys
import argparse, os
import torch
from torch.autograd import Variable
import re
import numpy
import copy
import time

import logging
logger = logging.getLogger("user_activity")

from Validation_BasicFunction import mode, Test_Set, readYUVFile, calculatePSNR

# pytorch模型的深度学习（在组委会提供的代码上进行了改写，去除读写文件的操作与耗时，保证每个模型的公平）
def net_forward(model, image, patch_size):

    patchs = get_patch_test (patch_size, image.shape[1], image.shape[0], patch_size, image)

    predict_patchs = []
    for i in range(len(patchs)):
        patch_input = patchs[i]
        patch_input = Variable(torch.from_numpy(patch_input).float()).view(1, -1, patch_input.shape[0], patch_input.shape[1])
        patch_input = patch_input.cuda()
        patch_output = model(patch_input).cpu()
        predict_patchs.append(copy.deepcopy(patch_output.data[0][0]))        
     
    res_patch = joint_patch(patch_size, image.shape[1], image.shape[0], patch_size, predict_patchs)
    res = res_patch.transpose(1, 0)

    return res


# 对图片进行评分的核心程序
def pytorch_validation(pytorchModels, patchSize):

    counter = 0
    psnr_accu = 0.0
    t = time.time()

    # 提取用户的model位置
    path_list = pytorchModels[0].split(sep='/')
    import_model_path = '/'.join(path_list[:-1])
    sys.path.append(import_model_path)

    # 设置在哪张GPU上运行
    torch.cuda.set_device(1)

    qp_list = [38, 45, 52]

    for ii in range(3):

        # 参数
        qp = qp_list[ii]

        # 用户提交的模型文件
        model_path = pytorchModels[ii]

        # 提取用户参赛方提交的作品文件
        model = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
        model = model.cuda()    # 使用GPU

        for f in os.listdir('./%s/%s_Q%d_yuv' % (Test_Set, mode, qp)):

            # 获取图片的序号、宽、高
            tmpl = re.split('\_|x|\.', f)
            i = int(tmpl[0])
            w = int(tmpl[1])
            h = int(tmpl[2])

            # 读取测试集的YUV文件
            (Yo, Uo, Vo) = readYUVFile('./%s/%s_yuv/%d.yuv' % (Test_Set, mode, i), w, h)
            (Yd, Ud, Vd) = readYUVFile('./%s/%s_Q%d_yuv/%s' % (Test_Set, mode, qp, f), w, h)

            # 深度学习
            Yr = net_forward(model, Yd, patchSize)
            Ur = net_forward(model, Ud, patchSize)
            Vr = net_forward(model, Vd, patchSize)
            
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
            

# ---------------------------------- pytorch 功能函数（直接来源于组委会提供的函数）----------------------------------

def get_patch_test(stride, width, height, patch_size, img):
    img = img.transpose(1, 0)
    num_patch_w = int((width - patch_size) / stride + 1)
    num_patch_h = int((height - patch_size) / stride + 1)
    s = stride
    cs = patch_size
    patches = []
    for i in range(0, num_patch_w):
        for j in range(0, num_patch_h):
            A = img[i * s:i * s + cs, j * s:j * s+ cs]
            patches.append(A)
    if not num_patch_w * s == width:
        for j in range(0, num_patch_h):
            w_bound = img[num_patch_w * s:width, j * s:j * s + cs]
            patches.append(w_bound)
    if not num_patch_h * s == height:
        for i in range(0, num_patch_w):
            h_bound = img[i * s:i * s + cs, num_patch_h * s:height]
            patches.append(h_bound)
    if not num_patch_h * s == height and not num_patch_w * s == width:
        bound = img[num_patch_w * s:width, num_patch_h * s:height]
        patches.append(bound)
    return patches


# 图像的拼接
def joint_patch(stride, width, height, patch_size, patchs):
    num_patch_w = int((width - patch_size) / stride + 1)
    num_patch_h = int((height - patch_size) / stride + 1)
    s = stride
    cs = patch_size
    img =numpy.zeros((width,height))
    k=0
    for i in range(0, num_patch_w):
        for j in range(0, num_patch_h):
            img[int(i * s): int(i * s + cs), int(j * s):int(j * s+ cs)]=patchs[k]
            k+=1	
    if not num_patch_w * s == width:
        for j in range(0, num_patch_h):
            img[num_patch_w * s:width,j*s:j * s+ cs]=patchs[k]
            k+=1	
    if not num_patch_h * s == height:
         for i in range(0, num_patch_w):
             img[i * s:i * s + cs, num_patch_h * s:height]=patchs[k]
             k+=1	 
    if not num_patch_h * s == height and not num_patch_w * s == width:
         img[num_patch_w * s:width, num_patch_h * s:height]=patchs[k]
    return img



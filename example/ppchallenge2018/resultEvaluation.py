#coding:utf-8

# 函数说明：解压缩用户提交的作品，对作品里面的文件进行分类，判别用户用的是哪个模型

import re
import time
import numpy
import sys
import os
import requests
import json
import logging
logger = logging.getLogger("user_activity")
from Validation_BasicFunction import Default_modelType, Default_patchSize, decompress
from Run_Caffe import caffe_validation
from Run_Pytorch import pytorch_validation
from Run_Keras import keras_validation


# 结果评分的主程序
def calculateHandle(model_path):

    # 提取用户id和文件名称
    user_id = model_path.split(sep='/')[-2]
    file_name = model_path.split(sep='/')[-1]

    # 将用户解压到指定文件夹中（文件夹名称：去除后缀名）
    folder_name = '.'.join(file_name.split(sep='.')[:-1])

    # 解压缩
    file_path = user_id + '/' + file_name
    folder_path = user_id + '/' + folder_name
    decompress(file_path, folder_path)

    # 对提交模型进行分类和评分
    try:
        succeed, acc, cost = classify_model(user_id, folder_name)
        acc = round (acc, 4)  # PSNR保留四位小数点
        cost = round (cost, 2)  # 耗时保留两位小数点
        logger.info('State = ', succeed)
        logger.info('PSNR = %.4f, Time Cost = %.2f' % (acc, cost))
        return succeed, acc, cost
    except Exception as e:
        logger.error('failed ...')
        logger.error(e)
        raise


# 判断每个文件的后缀名或者文件后半部分的信息
def classify_model(user_id, folder_name):

    #有的用户的压缩包里有4个文件，但是也有的用户压缩包里现有一个文件夹 -> 做一个判断
    # files_list：用户提交作品的文件列表（包括后缀名）
    cur_path = './%s/%s' % (user_id, folder_name)
    files = os.listdir(cur_path)
    sub_path = os.path.join(cur_path, files[0])

    # 循环进入文件夹，直到指定文件出现（防止用户设置了多重目录）
    while os.path.isdir(sub_path):
        cur_path = sub_path
        files = os.listdir(sub_path)
        sub_path = os.path.join(cur_path, files[0])
    files_list = files

    # 默认值，防止有参赛者没提交config.json
    modelType = Default_modelType
    patchSize = Default_patchSize

    # 获取config.json里面的文件
    for cur_file in files_list:
        if cur_file.startswith('config'):
            readmePath = os.path.join(cur_path, cur_file)
            logger.info('===========config %s '%(readmePath))
            f = open(readmePath, encoding='utf-8')
            jsonObj = json.loads(f.read())
            modelType = str(jsonObj['framework'])
            logger.info('modelType : %s' % modelType)

            # TODO：判断JSON里面是否有patch_size？            
            patchSize = int(jsonObj['patch_size'])
            logger.info('patchSize : %d' % patchSize)

            break
        
            
    # 打开我们事先设置好的workType.json
    fil = open('workType.json', encoding='utf-8')
    configfile = json.loads(fil.read())
    #logger.info(configfile)
    #logger.info(configfile['types'][0]['name'])


    # ------------------------------- 判别是否为Caffe模型 -------------------------------
    if modelType.__contains__(configfile['types'][0]['name']):

        # 判断caffe模型里面的文件是否完备、格式是否正确
        is_caffe, prototxt_file, model38_file, model45_file, model52_file = check_caffe(cur_path, files_list)

        # 提交正确
        if is_caffe:
            logger.info('This is caffe model.')
            caffeModels = list ([model38_file, model45_file, model52_file])
            try:               
                acc, cost = caffe_validation(prototxt_file, caffeModels, patchSize)
                return (True, acc, cost)
            except Exception as e:
                logger.error(e)
                raise
        else:
            logger.info('The files are not complete!!')
            return (False, 0, 0)


    # ------------------------------- 判别是否为Torch模型 -------------------------------
    if modelType.__contains__(configfile['types'][1]['name']):

        # 判断torch模型里面的文件是否完备、格式是否正确
        is_pytorch, model38_file, model45_file, model52_file = check_pytorch(cur_path, files_list)

        # 提交正确
        if is_pytorch:
            logger.info('This is torch model.')
            pytorchModels = list ([model38_file, model45_file, model52_file])
            try:
                acc, cost = pytorch_validation(pytorchModels, patchSize)
                return (True, acc, cost)
            except Exception as e:
                logger.error(e)
                raise    
        else:
            logger.info('The files are not complete!!')
            return (False, 0, 0)


    # ------------------------------- 判别是否为keras模型 -------------------------------
    if modelType.__contains__(configfile['types'][2]['name']):

        # 判断keras模型里面的文件是否完备、格式是否正确
        is_keras, model38_file, model45_file, model52_file = check_keras(cur_path, files_list)

        # 提交正确
        if is_keras:
            logger.info ('This is keras model.')
            kerasModels = list ([model38_file, model45_file, model52_file])
            try:                
                acc, cost = keras_validation(kerasModels, patchSize)
                return (True, acc, cost)
            except Exception as e:
                logger.error(e)
                raise   
        else:
            logger.info('The files are not complete!!')
            return (False, 0, 0)


    # ------------------------------- 不属于这3种模型 -------------------------------
    
    logger.info('Please check your files!!!!!')
    return (False, 0, 0)
        


# 判断用户在提交caffe模型时，所有文件是否提交完备、格式是否正确
def check_caffe(cur_path, files_list):
   
    is_prototxt = False   # 是否为prototxt文件
    is_model38 = False    # 是否为qp38.caffemodel文件
    is_model45 = False    # 是否为qp45.caffemodel文件
    is_model52 = False    # 是否为qp52.caffemodel文件

    for cur_file in files_list:
        if cur_file.endswith('.prototxt'):
            is_prototxt = True
            prototxt_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('38.caffemodel'):
            is_model38 = True
            model38_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('45.caffemodel'):
            is_model45 = True
            model45_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('52.caffemodel'):
            is_model52 = True
            model52_file = os.path.join(cur_path, cur_file)         

    if is_prototxt and is_model38 and is_model45 and is_model52:
        is_caffe = True
    else:
        is_caffe = False

    return is_caffe, prototxt_file, model38_file, model45_file, model52_file
        

# 判断用户在提交pytorch模型时，所有文件是否提交完备、格式是否正确
def check_pytorch(cur_path, files_list):

    is_model38 = False    # 是否为qp38.pth文件
    is_model45 = False    # 是否为qp45.pth文件
    is_model52 = False    # 是否为qp52.pth文件

    # model38_file、model45_file、model52_file为用户作品中的三个pth文件
    for cur_file in files_list:
        if cur_file.endswith('38.pth'):
            is_model38 = True
            model38_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('45.pth'):
            is_model45 = True
            model45_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('52.pth'):
            is_model52 = True
            model52_file = os.path.join(cur_path, cur_file)         

    # 如果三个pth文件齐全，则判定用户使用pytorch模型
    if is_model38 and is_model45 and is_model52:
        is_pytorch = True
    else:
        is_pytorch = False

    return is_pytorch, model38_file, model45_file, model52_file


# 判断用户在提交keras模型时，所有文件是否提交完备、格式是否正确
def check_keras(cur_path, files_list):

    is_model38 = False    # 是否为qp38.h5文件
    is_model45 = False    # 是否为qp45.h5文件
    is_model52 = False    # 是否为qp52.h5文件

    # model38_file、model45_file、model52_file为用户作品中的三个h5文件
    for cur_file in files_list:
        if cur_file.endswith('38.h5'):
            is_model38 = True
            model38_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('45.h5'):
            is_model45 = True
            model45_file = os.path.join(cur_path, cur_file)
        if cur_file.endswith('52.h5'):
            is_model52 = True
            model52_file = os.path.join(cur_path, cur_file)         

    # 如果三个h5文件齐全，则判定用户使用keras模型
    if is_model38 and is_model45 and is_model52:
        is_keras = True
    else:
        is_keras = False

    return is_keras, model38_file, model45_file, model52_file


import os
import shutil
import numpy
import sys


# 测试集名称
mode = 'test'

# 测试集的文件夹
Test_Set = 'chinamm2018_%s' % mode  

# 默认模型为caffe
Default_modelType = 'caffe'

# 将图像分块（注意运行baseline2的时候，ps太大的话，会报“out of memory”的错误！
# 默认值为256
Default_patchSize = 256


# 读取YUV文件，提取其中的参数
def readYUVFile(filename, width, height):
    with open(filename, 'rb') as rfile:
        Y = numpy.fromfile(rfile, 'uint8', width * height).reshape([height, width])
        wuv = width // 2
        huv = height // 2
        U = numpy.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
        V = numpy.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
    return (Y, U, V)


# 计算PSNR
# dtype=numpy.int64: 对于大分辨率的图像，MSE可能超出int32的表达范围
def calculatePSNR(orig, proc):

    diff = abs (numpy.int_(orig) - numpy.int_(proc))
    mse = numpy.sum(diff*diff, dtype=numpy.float64) / orig.size
    mse = abs(mse)

    if mse==0.0:
        mse=0.00001

    psnr = 10 * numpy.log10(255.0 * 255.0 / mse)
    return psnr


# 解压缩用户提交的作品
def decompress (file_path, folder_path):

    # 如果文件夹已经存在，删除里面所有内容；之后重新建立一个文件夹
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    # 提交的模型中只支持.zip和.rar两种格式
    if file_path.endswith('.zip'):
        cmd = 'unzip -o %s -d %s' % (file_path, folder_path)
        os.system(cmd)
        return

    if file_path.endswith('.rar'):
        cmd = 'rar x %s %s' % (file_path, folder_path)
        os.system(cmd)
        return



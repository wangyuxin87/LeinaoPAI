#coding:utf-8

import numpy as np
import logging
import argparse


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', default=10, type=int, choices=range(1, 1000),
                    help='times to read the whole dateset')
parser.add_argument('-f', '--filelist', default='./imagenet_val_files.txt',
                    help='list of files to read, stored in a txt file')
parser.add_argument('--step', default=1000, type=int,
                    help='log output when the number of files have read')


def main():
    global args
    args = parser.parse_args()

    # filelist = '/userhome/imagenet_val_files.txt'
    filelist = args.filelist
    epoch = args.epoch

    files = []
    with open(filelist, 'r') as fp:
        files = fp.readlines()
    total_files_number = len(files)

    for i in range(0, epoch):
        perm_index = np.random.permutation(total_files_number)
        count = 0
        for j in range(0, total_files_number):
            try:
                filename = files[perm_index[j]].strip()
                count += 1
                if count % args.step == 0:
                    logger.info('epoch/iter: %d/%d' % (i, count))
                with open(filename, 'rb') as fp:
                    fp.read()
            except:
                logger.error('read %s error' % filename)
                pass


if __name__ == '__main__':
    logger.info('try to read imagenet with random order')
    main()

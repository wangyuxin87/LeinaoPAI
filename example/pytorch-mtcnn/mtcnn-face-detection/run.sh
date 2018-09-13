#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0

python3 main_prepare.py configs/mtcnn_exp_p0.json
python3 main_train.py configs/mtcnn_exp_p0.json
python3 main_prepare.py configs/mtcnn_exp_r0.json
python3 main_train.py configs/mtcnn_exp_r0.json
python3 main_prepare.py configs/mtcnn_exp_o0.json
python3 main_train.py configs/mtcnn_exp_o0.json
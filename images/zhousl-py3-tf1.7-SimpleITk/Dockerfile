FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
 RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev && \
    apt-get install -y python3-tk unrar && \
    pip3 install numpy scipy matplotlib SimpleITK pillow pypng tifffile rarfile pandas tensorflow-gpu==1.7.0 h5py git && \
    pip3 install git+http://github.com/tflearn/tflearn.git && \
    pip3 install tflearn && \
    apt-get update -y && \
    apt-get install -y --allow-downgrades --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0 libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/* && \ 
    apt-get update -y

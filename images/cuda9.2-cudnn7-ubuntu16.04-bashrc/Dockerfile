FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH

# set a customizable and preservable ".bashrc" if necessary 
RUN sed -i '$a if [ -f /userhome/root/.bashrc ]; then\n  . /userhome/root/.bashrc\nfi' /root/.bashrc

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        openssh-client \
        openssh-server \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
EXPOSE 6006
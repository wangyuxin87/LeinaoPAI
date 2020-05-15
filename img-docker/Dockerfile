ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name mb python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=mb
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN conda install -c menpo opencv
RUN pip install ninja yacs cython matplotlib tqdm scipy shapely networkx pandas scikit-learn seaborn colour lmdb editdistance torchtext==0.4.0 spacy==2.2.2 tqdm dill numpy

RUN conda install pillow=5.2.0
# Install PyTorch 1.0 Nightly and OpenCV
RUN conda install -y pytorch=1.3.1 cudatoolkit=9.0 -c pytorch \
 && conda clean -ya
RUN conda install -y torchvision=0.2.1 -c pytorch \
 && conda clean -ya



# Install TorchVision master
#RUN git clone https://github.com/pytorch/vision.git \
#&& cd vision \
# && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

#install bdn
# cd /code \
# && RUN git clone https://github.com/Yuliang-Liu/Box_Discretization_Network.git \
# && cd Box_Discretization_Network
# && python setup.py build develop
# install PyTorch Detection
#RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
# && cd maskrcnn-benchmark \
# && python setup.py build develop

#WORKDIR /code/Box_Discretization_Network

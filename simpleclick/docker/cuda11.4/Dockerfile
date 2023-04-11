FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	    git \
	    curl \
        libglib2.0-0 \
        software-properties-common \
        python3-dev \
        python3-pip \
        python3-tk \
        firefox \
        libcanberra-gtk-module \
        nano

WORKDIR /tmp

RUN pip3 install --upgrade pip
RUN pip3 install matplotlib==3.3.4
RUN pip3 install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install setuptools==52.0.0
RUN pip3 install albumentations==0.5.2
RUN pip3 install PyYAML==6.0
RUN pip3 install easydict==1.9
RUN pip3 install tensorboard==2.8.0
RUN pip3 install opencv-python-headless==4.5.3.56
RUN pip3 install albumentations==0.5.2
RUN pip3 install mmcv==1.6.2
RUN pip3 install timm==0.6.11
RUN pip3 install Cython==0.29.32
RUN pip3 install protobuf==3.6.1
RUN pip3 install einops==0.6.0

RUN mkdir /work
WORKDIR /work
RUN chmod -R 777 /work && chmod -R 777 /root

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

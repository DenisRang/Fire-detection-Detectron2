FROM ubuntu:16.04
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/IRC_NAS/16803/l_openvino_toolkit_p_2020.4.287.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

COPY model.onnx .

RUN /bin/bash -c 'source $HOME/.bashrc; echo $HOME'
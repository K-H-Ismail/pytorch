#!/bin/bash

if [[ ${CUDNN_VERSION} == 8 ]]; then

    sudo apt-get update
    # also install ssh to avoid error of:
    # --------------------------------------------------------------------------
    # The value of the MCA parameter "plm_rsh_agent" was set to a path
    # that could not be found:
    #   plm_rsh_agent: ssh : rsh
    sudo apt-get install -y ssh
    sudo apt-get update && apt-get install -y --no-install-recommends libcudnn8=8.3.2.44-1+cuda11.5 libcudnn8-dev=8.3.2.44-1+cuda11.5 && apt-mark hold libcudnn8

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    #mkdir tmp_cudnn && cd tmp_cudnn
    #CUDNN_NAME="cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive"
    #curl -OLs  https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/${CUDNN_NAME}.tar.xz
    #tar xf ${CUDNN_NAME}.tar.xz
    #cp -a ${CUDNN_NAME}/include/* /usr/local/cuda/include/
    #cp -a ${CUDNN_NAME}/lib/* /usr/local/cuda/lib64/
    #cd ..
    #rm -rf tmp_cudnn
    #ldconfig
fi

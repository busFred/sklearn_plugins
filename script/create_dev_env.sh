#!/bin/bash
PROJECT_ROOT_PATH=${PWD}/..
# activate conda
source ~/anaconda3/bin/activate
# create virtual environment
conda create --name sklearn_plugins python=3.8.5 -y
conda activate sklearn_plugins
# install package
pip3 install numpy scipy scikit-learn matplotlib
pip3 install onnx onnxruntime-gpu
# pip3 install skl2onnx # requires 1.9.0 which is not yet available on pip3
pip3 install autopep8
# add import path
conda develop $PROJECT_ROOT_PATH/src
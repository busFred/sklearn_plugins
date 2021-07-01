#!/bin/bash
PROJECT_ROOT_PATH=${PWD}/..
# activate conda
source ~/anaconda3/bin/activate
# create virtual environment
conda create --name sklearn_plugins_test python=3.8.5 -y
conda activate sklearn_plugins_test
#!/bin/bash
mkdir -p model_repository
docker run --gpus all --detach -it --shm-size=256m --rm  -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.10-py3 tritonserver --model-repository=/models --model-control-mode=explicit
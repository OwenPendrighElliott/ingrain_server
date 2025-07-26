#!/bin/bash
mkdir -p model_repository
docker run --gpus all -dit --name triton_container --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:25.06-py3 tritonserver --model-repository=/models --model-control-mode=explicit
# docker run --gpus all -dit --name triton_container --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:25.06-py3 tritonserver --model-repository=/models --model-control-mode=explicit


# Welcome

Ingrain is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

For more detailed getting started instructions, please see the [getting started guide](getting_started/index.html).

The recommended way to run Ingrain locally is via Docker with a docker compose file to network with triton.

```yml
services:
  ingrain:
    image: owenpelliott/ingrain-server:latest
    container_name: ingrain
    ports:
      - "8686:8686"
      - "8687:8687"
    environment:
      TRITON_GRPC_URL: triton:8001
      MAX_BATCH_SIZE: 8
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
      - ${HOME}/.cache/huggingface:/app/model_cache/
  triton:
    image: nvcr.io/nvidia/tritonserver:25.04-py3
    container_name: triton
    runtime: nvidia # comment out if not using GPU
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    shm_size: "256m"
    command: >
      tritonserver
      --model-repository=/models
      --model-control-mode=explicit
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./model_repository:/models
    restart:
      always
```

To run without a GPU comment out the `runtime: nvidia` in the triton container.

Spin up the server with:

```bash
docker compose up
```

## Usage

The server is available via a REST API, there is also a Python client available.

```bash
pip install ingrain
```

## What does it do?

This server handles all the model loading, ONNX conversion, memory management, parallelisation, dynamic batching, input pre-processing, image handling, and other complexities of running a model in production. The API is very simple but lets you serve models in a performant manner.

Open CLIP models and sentence transformers are both converted to ONNX and served by Triton. The server can handle multiple models at once.

## How does it perform?

It retains all the performance of Triton. On 12 cores at 4.3 GHz with a 2080 SUPER 8GB card running in Docker using WSL2, it can serve `intfloat/e5-small-v2` to 500 clients at ~1050 QPS, or `intfloat/e5-base-v2` to 500 clients at ~860 QPS.

## How compatible is it?

Most models work out of the box, it is intractable to test every sentence transformers model and every CLIP models but most main architectures are tested and work. If you have a model that doesn't work, please open an issue.
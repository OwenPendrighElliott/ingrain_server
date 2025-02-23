
# Welcome

Ingrain is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

To use:
```bash
docker run --name ingrain_server -p 8686:8686 -p 8687:8687 --gpus all owenpelliott/ingrain-server:latest
```

To run without a GPU remove the `--gpus all` flag.

## Usage

The serve is available via a REST API, there is also a Python client available.

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
# Ingrain Server

> __This project is under active development, expect breaking changes between versions until v1.0.0.__

This is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

Ingrain server works in tandem with Triton to automate much of the process for serving OpenCLIP, sentence transformers, and timm models. 

The easiest way to get started is with a docker compose file, which will run Triton and the Ingrain server.

```yml
version: "3.9"

services:
  ingrain:
    image: owenpelliott/ingrain-server:latest
    container_name: ingrain
    ports:
      - "8686:8686"
      - "8687:8687"
    environment:
      TRITON_GRPC_URL: triton:8001
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
      - LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libtcmalloc.so.4
    shm_size: "1g"
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

## What does it do?

This server handles all the model loading, ONNX conversion, memory management, parallelisation, dynamic batching, input pre-processing, image handling, and other complexities of running a model in production. The API is very simple but lets you serve models in a performant manner.

Open CLIP models and sentence transformers are both converted to ONNX and served by Triton. The server can handle multiple models at once.

## How does it perform?

It retains all the performance of Triton. On 12 cores at 4.3 GHz with a 2080 SUPER 8GB card running in Docker using WSL2, it can serve `intfloat/e5-small-v2` to 100 clients at ~1310 QPS, or `intfloat/e5-base-v2` to 100 clients at ~1158 QPS.

## How compatible is it?

Most models work out of the box, it is intractable to test every sentence transformers model and every CLIP models but most main architectures are tested and work. If you have a model that doesn't work, please open an issue.

## Usage

The easiest way to get started is the use the optimised Python Client:

```bash
pip install ingrain
```

```python
import ingrain

ingrn = ingrain.Client()

model = ingrn.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = model.infer_text(text=["I am a sentence.", "I am another sentence.", "I am a third sentence."])

print(f"Processing Time (ms): {response['processingTimeMs']}")
print(f"Text Embeddings: {response['embeddings']}")
```

You can also have the embeddings automatically be returned as a numpy array:

```python
import ingrain

ingrn = ingrain.Client(return_numpy=True)

model = client.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = model.infer_text(text=["I am a sentence.", "I am another sentence.", "I am a third sentence."])

print(type(response['embeddings']))
```

### Example Requests and Responses

##### Loading a sentence transformer model `POST /load_sentence_transformer_model`:
```
{
    "name": "intfloat/e5-small-v2",
}
```

##### Loading a CLIP model `POST /load_clip_model`:
```
{
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k"
}
```

##### Inference request `POST /infer`:
```
{
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "text": ["I am a sentence.", "I am another sentence.", "I am a third sentence."],
    "image": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
}
```
Response
```
{
    "textEmbeddings": [
        [0.1, ..., 0.4],
        [0.5, ..., 0.8],
        [-0.2, ..., 0.3],
    ]
    "imageEmbeddings": [
        [0.1, ..., 0.4],
        [0.5, ..., 0.8],
    ],
    "processingTimeMs": 24.34
}
```

##### Get inference metrics `GET /metrics`:
Details omitted for brevity.
```
{
  "modelStats": [
    {
      "name": "ViT-B-32_laion2b_s34b_b79k_image_encoder",
      "version": "1",
      "inference_stats": {
        "success": {...},
        "fail": {...},
        "queue": {...},
        "compute_input": {...},
        "compute_infer": {...},
        "compute_output": {...},
        "cache_hit": {...},
        "cache_miss": {...}
      }
    },
    {
      "name": "ViT-B-32_laion2b_s34b_b79k_text_encoder",
      "version": "1",
      "inference_stats": {
        ...
      }
    },
    {
      "name": "intfloat_e5-small-v2",
      "version": "1",
      "inference_stats": {
        ...
      }
    }
  ]
}
```

##### Unloading a model `POST /unload_model`:
```
{
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k"
}
```

##### Delete a model `POST /delete_model`:
```
{
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k"
}
```

## Build Container Locally

Build the Docker image:

```bash
docker build -t ingrain-server .
```

Run the Docker container:

```bash
docker run --name ingrain_server -p 8686:8686 -p 8687:8687 --gpus all ingrain-server
```

## Performance test

You can run the benchmark script to test the performance of the server:

Install the python client:
```bash
pip install ingrain
```

Run the benchmark script:
```bash
python benchmark.py
```

It will output some metrics about the inference speed of the server.

```
{"message":"Model intfloat/e5-small-v2 is already loaded."}
Benchmark results:
Concurrent threads: 500
Requests per thread: 20
Total requests: 10000
Total benchmark time: 9.31 seconds
QPS: 1074.66
Mean response time: 0.3595 seconds
Median response time: 0.3495 seconds
Standard deviation of response times: 0.1174 seconds
Mean inference time: 235.5968 ms
Median inference time: 227.6743 ms
Standard deviation of inference times: 84.8669 ms
```

## Development Setup

Requires Docker and Python to be installed.

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Triton Inference Server

```bash
bash run_triton_server_dev.sh
```

### Run the FastAPI server

```bash
uvicorn inference_server:app --host 127.0.0.1 --port 8686 --reload
```

```bash
uvicorn model_server:app --host 127.0.0.1 --port 8687 --reload
```

### Testing

Install `pytest`:
    
```bash
pip install pytest
```

#### Unit tests

```bash
pytest
```

#### Integration tests and unit tests

```bash
pytest --integration
```

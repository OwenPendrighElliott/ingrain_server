# Ingrain Server

This is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

## What does it do?

This server handles all the model loading, tracing, ONNX conversion, memory management, parallelisation, dynamic batching, input pre-processing, image handling, and other complexities of running a model in production. The API is very simple but lets you serve models in a performant manner.

## How does it perform?

It retains all the performance of Triton. On 12 cores at 4.3 GHz with a 2080 SUPER 8GB card running in Docker using WSL2, it can serve `intfloat/e5-small-v2` to 500 clients at ~560 QPS, or `intfloat/e5-base-v2` to 500 clients at ~500 QPS.

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
    "text_embeddings": [
        [0.1, ..., 0.4],
        [0.5, ..., 0.8],
        [-0.2, ..., 0.3],
    ]
    "image_embeddings": [
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
  "model_stats": [
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

## Deployment Setup

Build the Docker image:

```bash
docker build -t ingrain-server .
```

Run the Docker container:

```bash
docker run --name ingrain_server -p 8686:8686 --gpus all ingrain-server
```

## Performance test

You can run the benchmark script to test the performance of the server:

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
Total benchmark time: 17.28 seconds
QPS: 578.56
Mean response time: 0.7786 seconds
Median response time: 0.6470 seconds
Standard deviation of response times: 0.5563 seconds
Median inference time: 9.0256 ms
Standard deviation of inference times: 3.0925 ms
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
uvicorn app:app --host 127.0.0.1 --port 8686 --reload
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

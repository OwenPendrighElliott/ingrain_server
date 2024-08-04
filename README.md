# Inference Server

This is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

## Deployment Setup

Build the Docker image:

```bash
docker build -t inference-server .
```

Run the Docker container:

```bash
docker run -d --rm -p 8686:8686 --gpus all inference-server
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
bash run_triton_server.sh
```

### Run the FastAPI server

```bash
uvicorn app:app --host 127.0.0.1 --port 8686 --reload
```




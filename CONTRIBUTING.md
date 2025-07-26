# Contributing to Ingrain Server

Ingrain Server welcomes contributions.

## Local Development Setup

### Triton

Ingrain Server uses the Triton Inference Server, for developement you can run it as its own container. To do this, you can use the provided script:

```bash
bash run_triton_server_dev.sh
```

You may need to remove `--gpus all` within the script if you don't have a GPU available, please don't commit this change if you do make it.

### Python 

Ingrain Server uses `uv` for dependency management. To setup the project, run:

```bash
uv sync --dev
```

The two FastAPI servers can then be started as follows:

```bash
uv run uvicorn --app-dir ingrain_inference_server inference_server:app --host 127.0.0.1 --port 8686 --reload
```

```bash
uv run uvicorn --app-dir ingrain_model_server model_server:app --host 127.0.0.1 --port 8687 --reload
```

## Testing

Ingrain Server uses `pytest` for testing. To run the unit tests, you can use:

```bash
uv run pytest
```

There are also integration tests which require the servers and triton to be running. You can run these tests with:

```bash
uv run pytest --integration
```

## Code Style Checks

All code checks should pass before raising a PR. You can run the checks locally using:

```bash
uv run black . --check
uv run ruff check
```
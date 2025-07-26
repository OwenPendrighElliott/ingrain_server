#!/bin/bash
set -e
# Run hello.sh
bash /app/hello.sh
# Start the inference server
uvicorn inference_server:app --host 0.0.0.0 --port 8686 --log-level="error" --workers=$(nproc)
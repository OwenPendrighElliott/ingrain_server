#!/bin/bash

# Print the hello message
bash hello.sh

# Start both Triton and Uvicorn in parallel
nohup tritonserver --model-repository=/app/model_repository --model-control-mode=explicit > /app/tritonserver.log 2>&1 & \
/app/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8686 --log-level="error" --workers 8

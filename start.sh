#!/bin/bash
# Print the hello message
bash /app/hello.sh
# Start Triton server in the background
nohup tritonserver --model-repository=/app/model_repository --model-control-mode=explicit > /app/tritonserver.log 2>&1 &
# Start two Uvicorn servers in parallel:
nohup uvicorn inference_server:app --host 0.0.0.0 --port 8686 --log-level="error" --workers $(nproc) > /app/inference_server.log 2>&1 &
nohup uvicorn model_server:app --host 0.0.0.0 --port 8687 > /app/model_server.log 2>&1 &
tail -f /app/*.log
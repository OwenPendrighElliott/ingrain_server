#!/bin/bash
set -e
# Run hello.sh
bash /app/hello.sh
# Start the model server
uvicorn model_server:app --host 0.0.0.0 --port 8687 --workers=1
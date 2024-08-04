#!/bin/bash
source /app/venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8686 --log-level "error" --workers 8

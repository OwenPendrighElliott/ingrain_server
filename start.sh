# #!/bin/bash
# shutdown() {
#     echo "Gracefully shutting down..."
#     pkill -SIGTERM -f tritonserver
#     pkill -SIGTERM -f "uvicorn inference_server:app"
#     pkill -SIGTERM -f "uvicorn model_server:app"
#     echo "All processes terminated. Exiting."
#     exit 0
# }
# trap shutdown SIGTERM SIGINT
# bash /app/hello.sh
# nohup tritonserver --model-repository=/app/model_repository --model-control-mode=explicit > /app/tritonserver.log 2>&1 &
# nohup uvicorn inference_server:app --host 0.0.0.0 --port 8686 --log-level="error" --workers $(nproc) > /app/inference_server.log 2>&1 &
# nohup uvicorn model_server:app --host 0.0.0.0 --port 8687 > /app/model_server.log 2>&1 &
#!/bin/bash
set -e
# Run hello.sh
echo "Running hello.sh..."
bash /app/hello.sh
# Start Supervisord
echo "Starting Supervisord..."
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf

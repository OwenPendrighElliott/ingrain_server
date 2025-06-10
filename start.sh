#!/bin/bash
set -e
# Run hello.sh
echo "Running hello.sh..."
bash /app/hello.sh
# Start Supervisord
echo "Starting Supervisord..."
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf

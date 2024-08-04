# Use the NVIDIA Triton server image as the base
FROM nvcr.io/nvidia/tritonserver:24.07-py3

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip python3-venv && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the required ports
EXPOSE 8686

# Create the model repository directory
RUN mkdir -p /app/model_repository

# Copy and make the inference server run script executable
COPY run_inference_server.sh /app/
RUN chmod +x /app/run_inference_server.sh

# Start Triton server and inference server
CMD tritonserver --model-repository=/app/model_repository --model-control-mode=explicit & \
    /app/run_inference_server.sh

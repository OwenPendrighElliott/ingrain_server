# Use the NVIDIA Triton server image as the base
FROM nvcr.io/nvidia/tritonserver:24.07-py3

# Remove unnecessary backends (keep only PyTorch and ONNX)
RUN rm -rf /opt/tritonserver/backends/dali \
           /opt/tritonserver/backends/fil \
           /opt/tritonserver/backends/identity \
           /opt/tritonserver/backends/python \
           /opt/tritonserver/backends/repeat \
           /opt/tritonserver/backends/square \
           /opt/tritonserver/backends/tensorflow \
           /opt/tritonserver/backends/tensorrt

# Install the required packages
RUN apt-get update && apt-get install -y python3-pip python3-venv

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /app/venv

# Copy only requirements.txt first to take advantage of Docker's cache
COPY requirements.txt /app

# Install Python dependencies
RUN . /app/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the required ports
EXPOSE 8686

# Create the model repository directory
RUN mkdir -p /app/model_repository

# Ensure the inference server run script is executable
RUN chmod +x /app/start.sh

# Start the app
CMD bash /app/start.sh


# # Stage 1: Build the Triton server with only PyTorch and ONNX backends
# FROM nvcr.io/nvidia/tritonserver:24.07-py3 AS triton_build

# # Remove unnecessary backends (keep only PyTorch and ONNX)
# RUN rm -rf /opt/tritonserver/backends/dali \
#            /opt/tritonserver/backends/fil \
#            /opt/tritonserver/backends/identity \
#            /opt/tritonserver/backends/python \
#            /opt/tritonserver/backends/repeat \
#            /opt/tritonserver/backends/square \
#            /opt/tritonserver/backends/tensorflow \
#            /opt/tritonserver/backends/tensorrt

# # Stage 2: Create a minimal image and copy only the necessary Triton files
# FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# # Install required dependencies
# RUN apt-get update && apt-get install -y python3-pip python3-venv && \
#     rm -rf /var/lib/apt/lists/*

# # Copy Triton from the first stage
# COPY --from=triton_build /opt/tritonserver /opt/tritonserver

# # Set environment variables for Triton
# ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# ENV PATH=/opt/tritonserver/bin:$PATH

# # Set the working directory
# WORKDIR /app

# # Create a virtual environment
# RUN python3 -m venv /app/venv

# # Copy only requirements.txt first to take advantage of Docker's cache
# COPY requirements.txt /app

# # Install Python dependencies in the virtual environment
# RUN . /app/venv/bin/activate && \
#     pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Copy the rest of the application code into the container
# COPY . /app

# # Expose the required Triton ports (default Triton ports)
# EXPOSE 8000 8001 8002

# # Create the model repository directory
# RUN mkdir -p /app/model_repository

# # Ensure the inference server run script is executable
# RUN chmod +x /app/start.sh

# # Start Triton Inference Server and the app
# CMD ["bash", "/app/start.sh"]

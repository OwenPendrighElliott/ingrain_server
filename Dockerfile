# Use the NVIDIA Triton server image as the base
FROM nvcr.io/nvidia/tritonserver:24.07-py3

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

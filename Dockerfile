# ARG to specify the base image
ARG BASE_IMAGE=owenpelliott/ingrain-gpu-base:amd64

# Use the base image specified in the build argument
FROM ${BASE_IMAGE}

# Install the required packages
RUN apt-get update && apt-get install -y python3-pip python3-venv

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to take advantage of Docker's cache
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the required ports
EXPOSE 8686
EXPOSE 8687

# Create the model repository directory
RUN mkdir -p /app/model_repository

# Ensure the inference server run script is executable
RUN chmod +x /app/start.sh

# Start the app
CMD ["bash", "/app/start.sh"]

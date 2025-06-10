# Use the base image specified in the build argument
FROM python:3.13.4-slim-bookworm

# Install the required packages
RUN apt-get update && apt-get install -y supervisor

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to take advantage of Docker's cache
COPY requirements.txt /app

RUN mkdir -p /app/model_cache/

ENV HF_HOME=/app/model_cache/

# Install Python dependencies
RUN pip install --upgrade pip --ignore-installed pip && \
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

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# Start the app
CMD ["bash", "/app/start.sh"]
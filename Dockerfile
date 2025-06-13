# Use the base image specified in the build argument
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install the required packages
RUN apt-get update && apt-get install -y supervisor

# Set the working directory
WORKDIR /app

RUN mkdir -p /app/model_cache/

ENV HF_HOME=/app/model_cache/

# Install Python dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the rest of the application code into the container
COPY . /app

# Create a virtual environment and install the dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

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
# Use CUDA 12.8 runtime as base image for lightweight deployment
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (without python3.12 first)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-pip \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.cargo/bin sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support and sm120 architecture support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other requirements from requirements.txt
RUN uv pip install --system -r requirements.txt

# Copy the entire application
COPY . /app

# Expose the port that the FastAPI server runs on
EXPOSE 8000

# Set the default command to run the web server
CMD ["python3", "webapp/server.py", "--port", "8000", "--no-debug"]

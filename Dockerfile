# Multi-stage build for smaller final image
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-pip \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Create working directory
WORKDIR /app

# Copy requirements file for installation first (for better caching)
COPY requirements.txt .

# PyTorch is already installed in the NGC base image, skip PyTorch installation

# Install requirements from requirements.txt using system python with uv
RUN uv pip install --system --break-system-packages -r requirements.txt

# Final stage - runtime image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install minimal runtime dependencies including OpenGL libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python is already configured in the NGC base image

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Create working directory
WORKDIR /app

# Copy the entire application after dependencies are installed
COPY . /app

# Initialize git submodules if they exist and checkout specified branch
RUN git submodule update --init --recursive || true && \
    git submodule foreach --recursive 'git checkout $branch || git checkout $sha1 || true'

# Force reinstall numpy after submodules to ensure we keep our version
RUN uv pip install --system --break-system-packages numpy==1.26.0 --force-reinstall

# Copy the environment file (Do this last)
COPY .env .
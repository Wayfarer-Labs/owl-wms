<<<<<<< HEAD
# Use CUDA 12.8 runtime as base image for lightweight deployment
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
=======
# Multi-stage build for smaller final image
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS builder
>>>>>>> uncond

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
<<<<<<< HEAD
ENV PYTHONPATH=/app

# Install system dependencies (without python3.12 first)
=======

# Install system dependencies 
>>>>>>> uncond
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-pip \
    git \
    software-properties-common \
<<<<<<< HEAD
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
=======
    && add-apt-repository ppa:deadsnakes/ppa \
>>>>>>> uncond
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Set Python 3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.cargo/bin sh
ENV PATH="/root/.cargo/bin:${PATH}"
=======
# Install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
>>>>>>> uncond

# Create working directory
WORKDIR /app

<<<<<<< HEAD
# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support and sm120 architecture support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other requirements from requirements.txt
RUN uv pip install --system -r requirements.txt

RUN git submodule update --init --recursive

# Copy the entire application
COPY . /app

# Expose the port that the FastAPI server runs on
EXPOSE 8000

# Set the default command to run the web server
CMD ["python3", "webapp/server.py", "--port", "8000", "--no-debug"]
=======
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
>>>>>>> uncond

FROM nvcr.io/nvidia/physicsnemo/physicsnemo:25.06

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y bash \
    build-essential \
    procps \
    wget \
    curl \
    git \
    ca-certificates \
    less \
    vim \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Bring in changes from outside container to /tmp (assumes my-pytorch-modifications.patch is in same directory as Dockerfile)
COPY torch_nightly.patch /tmp

# Change working directory to PyTorch source path
WORKDIR /opt/pytorch/pytorch

# Apply modifications
RUN patch -p1 < /tmp/torch_nightly.patch

# Rebuild PyTorch
RUN cd pytorch && \
    USE_CUPTI_SO=1 \
    USE_KINETO=1 \
    CMAKE_PREFIX_PATH="/usr/local" \
    NCCL_ROOT="/usr" \
    USE_SYSTEM_NCCL=1 \
    USE_UCC=1 \
    USE_SYSTEM_UCC=1 \
    UCC_HOME="/opt/hpcx/ucc" \
    # UCC_DIR is for PyTorch to find ucc-config.cmake
    UCC_DIR="/opt/hpcx/ucc/lib/cmake/ucc" \
    UCX_HOME="/opt/hpcx/ucx" \
    UCX_DIR="/opt/hpcx/ucx/lib/cmake/ucx" \
    CFLAGS='-fno-gnu-unique' \
    DEFAULT_INTEL_MKL_DIR="/usr/local" \
    INTEL_MKL_DIR="/usr/local" \
    python setup.py install \
    && python setup.py clean

WORKDIR /app
COPY . .

RUN python -m pip install -r requirements.txt

RUN git submodule update --init --recursive

CMD ["/bin/bash"]

The text leading up to this was:
    --------------------------
    |diff --git a/torch/onnx/ops/_symbolic_impl.py b/torch/onnx/ops/_symbolic_impl.py
    |index 7dd1370720a..4876612ad97 100644
    |--- a/torch/onnx/ops/_symbolic_impl.py
    |+++ b/torch/onnx/ops/_symbolic_impl.py
    --------------------------
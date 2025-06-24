# owl_wms
Basic world models

## Docker setup:
```
docker build -t owl_wms .

docker run --gpus all -it \
  -v $HOME/.gitconfig:/root/.gitconfig:ro \
  -v $HOME/.ssh:/root/.ssh:ro \
  --shm-size 1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  owl_wms /bin/bash
```

## Use Pytorch Nightly Build to Patch Pytorch Container via NGC:

PyTorch container reference commit:
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html

79aa17489c3fc5ed6d5e972e9ffddf73e6dd0a5c is the torch commit ID for pytorch 25.04 container.

git format-patch 79aa17489c3fc5ed6d5e972e9ffddf73e6dd0a5c..HEAD --stdout > torch_nightly.patch

**From NVIDIA docker-examples**:
"
Apply patches to the source code in NVIDIA's PyTorch container image and to rebuild PyTorch.
The RUN command included below will rebuild PyTorch in the same way as it was built in the original image.
"

**Dockerfile changes**:
# Bring in changes from outside container to /tmp (assumes my-pytorch-modifications.patch is in same directory as Dockerfile)
COPY my-pytorch-modifications.patch /tmp

# Change working directory to PyTorch source path
WORKDIR /opt/pytorch

# Apply modifications
RUN patch -p1 < /tmp/my-pytorch-modifications.patch

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



USE_CUPTI_SO=1 USE_KINETO=1 CMAKE_PREFIX_PATH="/usr/local" NCCL_ROOT="/usr" USE_SYSTEM_NCCL=1 USE_UCC=1 USE_SYSTEM_UCC=1 UCC_HOME="/opt/hpcx/ucc" UCC_DIR="/opt/hpcx/ucc/lib/cmake/ucc" UCX_HOME="/opt/hpcx/ucx" UCX_DIR="/opt/hpcx/ucx/lib/cmake/ucx" CFLAGS='-fno-gnu-unique' DEFAULT_INTEL_MKL_DIR="/usr/local" INTEL_MKL_DIR="/usr/local" python setup.py install

USE_CUPTI_SO=1 USE_KINETO=1 CMAKE_PREFIX_PATH="/usr/local" NCCL_ROOT="/usr" USE_SYSTEM_NCCL=1 USE_UCC=1 USE_SYSTEM_UCC=1 UCC_HOME="/opt/hpcx/ucc" UCC_DIR="/opt/hpcx/ucc/lib/cmake/ucc" UCX_HOME="/opt/hpcx/ucx" UCX_DIR="/opt/hpcx/ucx/lib/cmake/ucx" CFLAGS='-fno-gnu-unique' DEFAULT_INTEL_MKL_DIR="/usr/local" INTEL_MKL_DIR="/usr/local" python setup.py clean

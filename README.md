# owl_wms
Basic world models

## Training

### Single Node Setup

```bash
git clone https://github.com/wayfarer-labs/owl-wms
cd owl-wms
pip install -r requirements.txt
git submodule init
git submodule update
cd owl-vaes
git switch main
pip install -r requirements.txt
cd ..
# Configure WandB
wandb login
```

Then run training:
```bash
# Single GPU
python train.py --config_path configs/basic.yml

# Multi-GPU (single node)
torchrun --nproc_per_node=8 train.py --config_path configs/basic.yml
```

### Multi-Node Training

For multi-node distributed training:
```bash
torchrun --nproc_per_node=8 --nodes=2 train.py --config_path configs/av_v5_8x8_weak.yml
```

Or with SkyPilot:
```bash
sky launch --infra kubernetes --gpus H200:8 --num-nodes 2 --name <label> skypilot/config.yaml
```

## Docker Build & Deploy (For multinode training)

### Setup
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Configure your Docker registry settings in `.env`:
   ```bash
   # Docker Registry Configuration
   PROJECT_ID=your-project-id
   REGISTRY=us-central1-docker.pkg.dev
   REPOSITORY=your-repository
   IMAGE_NAME=your-image-name
   DEFAULT_TAG=latest
   
   # Local build configuration
   LOCAL_REGISTRY=us-central1-docker.pkg.dev
   LOCAL_PROJECT=your-local-project
   LOCAL_REPOSITORY=your-local-repo
   ```

### Build and Deploy
```bash
# Build, tag, and push with default tag
./build_and_push.sh

# Build, tag, and push with custom tag
./build_and_push.sh v1.0.0
```

The script will:
1. Build the Docker image locally
2. Tag it for your remote registry
3. Push to the configured registry
4. Update `skypilot/config.yaml` with the new image tag

## Multi-Node Training with SkyPilot

### Setup
1. Edit `skypilot/config.yaml` to specify your training configuration:
   ```yaml
   # Change this line to point to your config file
   train.py --config_path configs/YOUR_CONFIG.yml
   ```

2. Optionally adjust the number of nodes and GPU type:
   ```yaml
   resources:
     accelerators: H200:8  # 8 H200s per node
   num_nodes: 2            # Number of nodes
   ```

### Prerequisites
1. Make sure you're authenticated with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. Make sure you set your Project ID for google cloud.

### Launch Training
```bash
# Build and push your container
./build_and_push.sh

# Launch multi-node training on SkyPilot
sky launch skypilot/config.yaml

# Check job status
sky status

# View logs
sky logs <cluster_name>
```


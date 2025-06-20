import argparse
import os

from owl_wms.configs import Config
from owl_wms.trainers import get_trainer_cls
from owl_wms.utils.ddp import cleanup, setup

if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--config_path", "configs/self_forcing.yaml"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)

    global_rank, local_rank, world_size = setup()

    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train, cfg.wandb, cfg.model, global_rank, local_rank, world_size
    )

    trainer.train()
    cleanup()
    # nohup torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --config_path configs/self_forcing.yaml &

"""
Base class for any trainer
"""

import torch
import wandb
import os
import torch.distributed as dist

class BaseTrainer:
    def __init__(
        self,
        train_cfg, logging_cfg, model_cfg,
        global_rank = 0, local_rank = 0, world_size = 1
    ):
        self.rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.train_cfg = train_cfg
        self.logging_cfg = logging_cfg
        self.model_cfg = model_cfg

        self.device = f'cuda:{self.local_rank}'

        if self.logging_cfg is not None and self.rank == 0:
            log = self.logging_cfg
            # Login to wandb with API key from environment
            wandb_api_key = os.getenv('WANDB_API_KEY')
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            else:
                raise Exception("WANDB key not found, did you correctly load your .env file?")
            
            wandb.init(
                project = log.project,
                entity = log.name,
                name = log.run_name,
                config = {
                    'train' : train_cfg,
                    'model' : model_cfg
                }
            )
        
    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def get_module(self, ema = False):
        if self.world_size == 1:
            if ema:
                return self.ema.ema_model
            else:
                return self.model
        else:
            if ema:
                return self.ema.ema_model.module
            else:
                return self.model.module
    
    def save(self, save_dict):
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok = True)
        fp = os.path.join(self.train_cfg.checkpoint_dir, f"step_{self.total_step_counter}.pt")

        torch.save(save_dict, fp)

        if 'ema' in save_dict and getattr(self.train_cfg, 'output_path', None) is not None:
            out_d = save_dict['ema']
            prefix = "ema_model.module." if self.world_size > 1 else "ema_model."
            out_d = {k[len(prefix):]: v for k, v in out_d.items() if k.startswith(prefix)}
            os.makedirs(self.train_cfg.output_path, exist_ok = True)
            torch.save(out_d, os.path.join(self.train_cfg.output_path, f"step_{self.total_step_counter}.pt"))
    
    def load(self, path):
        return torch.load(path, map_location=f'cuda:{self.local_rank}',weights_only=False)

            
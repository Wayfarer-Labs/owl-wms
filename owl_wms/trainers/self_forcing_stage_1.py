import torch
from ema_pytorch import EMA
import wandb
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import einops as eo
from copy import deepcopy

from .base import BaseTrainer

from ..utils import freeze, unfreeze, Timer, find_unused_params, versatile_load
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb, to_wandb_av
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn

class CausalDistillationTrainer(BaseTrainer):
    """
    Causal Distillation Trainer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):  
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id

        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False

        self.model = get_model_cls(model_id)(student_cfg)
        self.teacher = get_model_cls(model_id)(teacher_cfg)

        self.teacher.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        self.model.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        freeze(self.teacher)

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0
        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        self.audio_decoder = get_decoder_only(
            self.train_cfg.audio_vae_id,
            self.train_cfg.audio_vae_cfg_path,
            self.train_cfg.audio_vae_ckpt_path
        )

        freeze(self.decoder)
        freeze(self.audio_decoder)

        #self.teacher.core = torch.compile(self.teacher.core, mode = 'max-autotune', fullgraph = True, dynamic = False)

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)
    
    def load(self):
        has_ckpt = False
        try:
            if self.train_cfg.resume_ckpt is not None:
                save_dict = super().load(self.train_cfg.resume_ckpt)
                has_ckpt = True
        except:
            print("Error loading checkpoint")
        
        if not has_ckpt:
            return

        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model and ema
        self.model = self.model.cuda().train()        
        self.teacher = self.teacher.cuda().eval().bfloat16()

        if self.world_size > 1:
            self.model = DDP(self.model)

        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.audio_decoder = self.audio_decoder.cuda().eval().bfloat16()

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        audio_decode_fn = make_batched_audio_decode_fn(self.audio_decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(
            self.model,
            beta = 0.999,
            update_after_step = 0,
            update_every = 1
        )

        def get_ema_core():
            if self.world_size > 1:
                return self.ema.ema_model.module.core
            else:
                return self.ema.ema_model.core

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda',torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')
        
        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for batch_vid, batch_audio, batch_mouse, batch_btn, cfg_mask in loader:
            batch_vid = batch_vid.cuda().bfloat16() / self.train_cfg.vae_scale
            batch_audio = batch_audio.cuda().bfloat16() / self.train_cfg.audio_vae_scale
            batch_mouse = batch_mouse.cuda().bfloat16()
            batch_btn = batch_btn.cuda().bfloat16()
            cfg_mask = cfg_mask.cuda()

            with ctx:
                # Student causal predictions
                student_out = self.model(batch_vid,batch_audio,batch_mouse,batch_btn,has_controls=cfg_mask, return_dict=True)
                with torch.no_grad():
                    teacher_video_pred, teacher_audio_pred = self.teacher.core(
                        student_out['lerpd_video'],
                        student_out['lerpd_audio'],
                        student_out['ts'],
                        batch_mouse,
                        batch_btn,
                        student_out['cfg_mask'],
                    )
                
                video_loss = F.mse_loss(student_out['pred_video'], teacher_video_pred)
                audio_loss = F.mse_loss(student_out['pred_audio'], teacher_audio_pred)
                loss = 0.5 * (video_loss + audio_loss) / accum_steps

            self.scaler.scale(loss).backward()
            metrics.log('loss', loss)

            local_step += 1
            if local_step % accum_steps == 0:
                # Updates
                if self.train_cfg.opt.lower() != "muon":
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.opt)
                self.opt.zero_grad(set_to_none=True)

                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()
                self.ema.update()

                # Do logging
                with torch.no_grad():
                    wandb_dict = metrics.pop()
                    wandb_dict['time'] = timer.hit()
                    timer.reset()

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        with ctx, torch.no_grad():
                            n_samples = self.train_cfg.n_samples
                            samples, audio, sample_mouse, sample_button = sampler(
                                get_ema_core(),
                                batch_vid[:n_samples],
                                batch_audio[:n_samples], 
                                batch_mouse[:n_samples],
                                batch_btn[:n_samples],
                                decode_fn,
                                audio_decode_fn,
                                self.train_cfg.vae_scale,
                                self.train_cfg.audio_vae_scale
                            )
                            if self.rank == 0:
                                video = to_wandb_av(samples, audio, sample_mouse, sample_button)
                                wandb_dict['samples'] = video
                        
                    if self.rank == 0:
                        wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    if self.rank == 0:
                        self.save()
                    
                self.barrier()

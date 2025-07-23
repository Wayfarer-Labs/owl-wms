import torch
from tqdm import tqdm
from ema_pytorch import EMA
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_av
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn

class AVRFTTrainer(BaseTrainer):
    """
    Trainer for rectified flow transformer

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
        self.model = get_model_cls(model_id)(self.model_cfg)

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
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            has_ckpt = True
        else:
            print("Failed to load checkpoint")
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
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
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
        #torch.compile(self.ema.ema_model.module.core if self.world_size > 1 else self.ema.ema_model.core, dynamic=False, fullgraph=True)

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
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.n_samples, **self.train_cfg.sample_data_kwargs)
        sample_loader = iter(sample_loader)

        if self.train_cfg.data_id == "cod_s3_mixed":
            loader.dataset.sleep_until_queues_filled()
            self.barrier()
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm(loader, total=len(loader), disable=self.rank != 0, desc=f"Epoch: {epoch}"):

                batch_vid, batch_audio, batch_mouse, batch_btn = [t.cuda().bfloat16() for t in batch]
                batch_vid = batch_vid / self.train_cfg.vae_scale
                batch_audio = batch_audio / self.train_cfg.audio_vae_scale

                with ctx:
                    loss = self.model(batch_vid,batch_audio,batch_mouse,batch_btn) / accum_steps

                self.scaler.scale(loss).backward()

                metrics.log('diffusion_loss', loss)

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    if self.train_cfg.opt.lower() != "muon":
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

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

                        # Sampling commented out for now
                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx, torch.no_grad():

                                vid_for_sample, aud_for_sample, mouse_for_sample, btn_for_sample = next(sample_loader)
                                n_samples = self.train_cfg.n_samples
                                samples, audio, sample_mouse, sample_button = sampler(
                                    get_ema_core(),
                                    vid_for_sample.bfloat16().cuda() / self.train_cfg.vae_scale,
                                    aud_for_sample.bfloat16().cuda() / self.train_cfg.audio_vae_scale,
                                    mouse_for_sample.bfloat16().cuda(),
                                    btn_for_sample.bfloat16().cuda(),
                                    decode_fn,
                                    audio_decode_fn,
                                    self.train_cfg.vae_scale,
                                    self.train_cfg.audio_vae_scale
                                ) # -> [b,n,c,h,w]
                                if self.rank == 0:
                                    wandb_av_out = to_wandb_av(samples, audio, sample_mouse, sample_button)
                                    if len(wandb_av_out) == 3:
                                        video, depth_gif, flow_gif = wandb_av_out
                                        wandb_dict['samples'] = video
                                        wandb_dict['depth_gif'] = depth_gif
                                        wandb_dict['flow_gif'] = flow_gif
                                    else:
                                        video = wandb_av_out
                                        wandb_dict['samples'] = video

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()

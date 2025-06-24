import torch
from ema_pytorch import EMA
import wandb
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import einops as eo
from copy import deepcopy

from .base import BaseTrainer

from owl_wms.utils import freeze, unfreeze, Timer, find_unused_params, versatile_load
from owl_wms.schedulers import get_scheduler_cls
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.data import get_loader
from owl_wms.utils.logging import LogHelper, to_wandb
from owl_wms.muon import init_muon
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class CausVidTrainer(BaseTrainer):
    """
    CausVid Trainer

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
        self.score_real = get_model_cls(model_id)(teacher_cfg)

        self.score_real.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        self.score_fake = deepcopy(self.score_real)

        freeze(self.score_real)

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.s_fake_opt = None
        self.scheduler = None
        self.s_fake_scaler = None
        self.scaler = None

        self.total_step_counter = 0
        self.decoder = get_decoder_only()
        freeze(self.decoder)

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'score_fake': self.score_fake.state_dict(),
            's_fake_opt': self.s_fake_opt.state_dict(),
            's_fake_scaler': self.s_fake_scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)
    
    def load_causal(self):
        has_ckpt = False
        try:
            if self.train_cfg.resume_ckpt is not None:
                save_dict = super().load_causal(self.train_cfg.resume_ckpt)
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
        self.score_fake.load_state_dict(save_dict['score_fake'])
        self.s_fake_opt.load_state_dict(save_dict['s_fake_opt'])
        self.s_fake_scaler.load_state_dict(save_dict['s_fake_scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model and ema
        self.model = self.model.cuda().train()        
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.score_real = self.score_real.cuda().eval().bfloat16()
        self.score_fake = self.score_fake.cuda().train()

        if self.world_size > 1:
            self.model = DDP(self.model)
            self.score_fake = DDP(self.score_fake)

        freeze(self.decoder)
        freeze(self.score_real)

        #torch.compile(self.score_real, dynamic = False)

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(
            self.model,
            beta = 0.999,
            update_after_step = 0,
            update_every = 1
        )
        # Hard coded stuff, probably #TODO figure out where to put this?
        self.update_ratio = 5
        self.cfg_scale = 1.3

        def get_ema_core():
            if self.world_size > 1:
                return self.ema.ema_model.module.core
            else:
                return self.ema.ema_model.core

        # Don't use MUON pls
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)
        self.s_fake_opt = getattr(torch.optim, self.train_cfg.opt)(self.score_fake.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Scaler
        self.s_fake_scaler = torch.amp.GradScaler()
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda',torch.bfloat16)

        self.load_causal()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')
        
        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)()

        # Simplifiying assumptions: data will never stop iter, no grad accum

        def sample_from_gen(vid, mouse, btn):  # Returns real video, cause noise - velocity
            model_out = self.model(vid, mouse, btn, return_dict = True)
            ts = model_out['ts'][:,None,None,None] # [b,n,c,h,w]
            lerpd = model_out['lerpd'] # [b,n,c,h,w]
            pred = model_out['pred'] # [b,n,c,h,w]

            samples = lerpd - pred*ts
            return samples

        def get_dmd_loss(vid, # sampled from student, real videos and not velocities
                         mouse,
                         btn,):
            s_real_fn = self.score_real.core
            s_fake_fn = self.score_fake.module.core

            with torch.no_grad():
                b,n,c,h,w = vid.shape
                ts = torch.randn(b,n,device=vid.device,dtype=vid.dtype).sigmoid()
                z = torch.randn_like(vid)
                ts_exp = ts[:,:,None,None,None]
            
            # NOTE SAMI: looks like rectified flow (linear interpolation)
            # ----- TEACHER CFG+Rectified Flow
            # keep track of the noise. "z" is only used in lerpd, so we can ignore the noise from q-sample-fn
            lerpd = vid * (1. - ts_exp) + z * ts_exp # noisy_causal_clip 

            null_mouse = torch.zeros_like(mouse)
            null_btn = torch.zeros_like(btn)

            s_real_uncond = s_real_fn(lerpd, ts, null_mouse, null_btn)
            s_real_cond = s_real_fn(lerpd, ts, mouse, btn)
            s_real = s_real_uncond + self.cfg_scale * (s_real_cond - s_real_uncond) # -- correct so far
            # ----- Critic Rectified Flow, no CFG needed?
            s_fake = s_fake_fn(lerpd, ts, mouse, btn) # critic.core(noised_clip), no cfg, correct so far.

            grad = (s_fake - s_real)

            # Normalizer? 
            p_real = (vid - s_real)
            normalizer = torch.abs(p_real).mean(dim=[1,2,3,4],keepdim=True)
            grad = grad / (normalizer + 1.0e-6)

            grad = torch.nan_to_num(grad)
            dmd_loss = 0.5 * F.mse_loss(vid.double(), vid.double() - grad.double())
            # ^ simplify to 0.5 * 2 * (vid - vid + grad) = grad, neat!
            return dmd_loss
        
        def optimizer_step(loss, model, scaler, optimizer):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

        loader = iter(loader)
        while True:
            # -- SAMI: This is thecritic update step, because we freeze the causal student and unfreeze the critic
            freeze(self.model)
            unfreeze(self.score_fake)
            for _ in range(self.update_ratio):
                batch_vid, batch_mouse, batch_btn = next(loader)
                with ctx:
                    with torch.no_grad():
                        samples = sample_from_gen(batch_vid, batch_mouse, batch_btn)
                        # -- SAMI: This calculates RF Loss: F.mse_loss(denoise(apply_noise(randn, sample)) (randn - sample))
                    # this is gamerft.fwd which is mse(pred_velocity, noise - samples),
                    # pred_velocity comes from gamerft.core(noise_video,). NOTE: This looks right on my end
                    s_fake_loss = self.score_fake(samples, batch_mouse, batch_btn) # this loss is mse(velocity_pred, noise - video)

                optimizer_step(s_fake_loss, self.score_fake, self.s_fake_scaler, self.s_fake_opt)

            metrics.log('s_fake_loss', s_fake_loss)
            # -- SAMI: This is the causal update step, because we unfreeze the causal student and freeze the critic
            unfreeze(self.model)
            freeze(self.score_fake)
        
            batch_vid, batch_mouse, batch_btn = next(loader)
            with ctx:
                samples = sample_from_gen(batch_vid, batch_mouse, batch_btn) # autoreg rollout *frames* not velocities
                dmd_loss = get_dmd_loss(samples, batch_mouse, batch_btn)
                metrics.log('dmd_loss', dmd_loss)
                
            optimizer_step(dmd_loss, self.model, self.scaler, self.opt)
            self.ema.update()

            with torch.no_grad():
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with ctx, torch.no_grad():
                        n_samples = self.train_cfg.n_samples
                        samples, sample_mouse, sample_button = sampler(
                            get_ema_core(),
                            batch_vid[:n_samples],
                            batch_mouse[:n_samples],
                            batch_btn[:n_samples],
                            decode_fn = decode_fn,
                            scale=self.train_cfg.vae_scale
                        ) # -> [b,n,c,h,w]
                        if self.rank == 0: wandb_dict['samples'] = to_wandb(samples, sample_mouse, sample_button)
                    
                if self.rank == 0:
                    wandb.log(wandb_dict)

            self.total_step_counter += 1
            if self.total_step_counter % self.train_cfg.save_interval == 0:
                if self.rank == 0:
                    self.save()
                
            self.barrier()
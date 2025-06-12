from owl_wms.models import get_model_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only
from owl_wms.configs import Config
from owl_wms.data import get_loader

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

class AVPipeline:
    def __init__(cfg_path="configs/av.yml", ckpt_path="av_dfot_35k_ema_200m.pt"):
        cfg = Config.from_yaml(cfg_path)
        model_cfg = cfg.model
        train_cfg = cfg.train
        
        model = get_model_cls(model_cfg.model_id)(model_cfg).core
        model.load_state_dict(ckpt_path)

        frame_decoder = get_decoder_only(
            None,
            train_cfg.vae_cfg_path,
            train_cfg.vae_ckpt_path
        )

        audio_decoder = get_decoder_only(
            None,
            train_cfg.audio_vae_cfg_path,
            train_cfg.audio_vae_ckpt_path
        )

        frame_scale = train_cfg.vae_scale
        audio_scale = train_cfg.audio_vae_scale

        self.history_buffer = None
        self.audio_buffer = None
        self.mouse_buffer = None
        self.button_buffer = None

        loader = get_loader(
            "cod_s3_audio",
            1,
            window_length=30,
            bucket_name='cod-data-latent-360x640to4x4'
        )
        self.loader = iter(loader)

        self.alpha = 0.2
        self.cfg = 1.3
        self.sampling_steps = 10

        torch.compile(self.model)
        torch.compile(self.frame_decoder)
        torch.compile(self.audio_decoder)

        self.audio_f = 735

    def init_buffers(self):
        self.history_buffer,self.audio_buffer,self.mouse_buffer,self.button_buffer=next(self.loader)
        # [1,n,c,h,w], [1,n,c], [1,n,2], [1,n,11]

        self.history_buffer = self.history_buffer / self.frame_scale
        self.audio_buffer = self.audio_buffer / self.audio_scale

    @torch.no_grad()
    def __call__(self, new_mouse, new_btn):
        # [2,] float and [11,] bool
        noised_history = zlerp(self.history_buffer[:,1:], self.alpha)
        noised_audio = zlerp(self.audio_buffer[:,1:], self.alpha)

        noised_history = torch.cat([noised_history, torch.randn_like(noised_history[:,0:1])], dim = 1)
        noised_audio = torch.cat([noised_audio, torch.randn_like(noised_audio[:,0:1])], dim = 1)

        new_mouse = new_mouse[None,None,:]
        new_btn = new_btn[None,None,:]

        self.mouse_buffer = torch.cat([self.mouse_buffer[:,1:],new_mouse],dim=1)
        self.button_buffer = torch.cat([self.button_buffer[:,1:],new_btn],dim=1)

        dt = 1. / self.sampling_steps

        x = noised_history
        a = noised_audio
        ts = torch.ones_like(noised_history[:,:,0,0,0])
        ts[:,:-1] = self.alpha

        mouse_batch = torch.cat([self.mouse_buffer, torch.zeros_like(mouse)], dim=0) 
        btn_batch = torch.cat([self.button_buffer, torch.zeros_like(btn)], dim=0)
        for _ in range(self.sampling_steps):
            x_batch = torch.cat([x, x], dim=0)
            a_batch = torch.cat([a, a], dim=0)
            ts_batch = torch.cat([ts, ts], dim=0)

            pred_vid_batch, pred_audio_batch = model(x_batch,a_batch,ts_batch,mouse_batch,btn_batch)

            cond_pred_video, uncond_pred_video = pred_video_batch.chunk(2)
            cond_pred_audio, uncond_pred_audio = pred_audio_batch.chunk(2)

            pred_video = uncond_pred_video + self.cfg_scale * (cond_pred_video - uncond_pred_video)
            pred_audio = uncond_pred_audio + self.cfg_scale * (cond_pred_audio - uncond_pred_audio)

            x[:,-1] = x[:,-1] - dt * pred_video[:,-1]
            a[:,-1] = a[:,-1] - dt * pred_audio[:,-1]
            ts[:,-1] = ts[:,-1] - dt
        
        new_frame = x[:,-1:] # [1,1,c,h,w]
        new_audio = audio[:,-1:] # [1,1,c]

        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1)
        self.audio_buffer = torch.cat([self.audio_buffer[:,1:], new_audio], dim=1)

        x_to_dec = new_frame[0] * self.image_scale
        a_to_dec = self.audio_buffer * self.audio_scale

        frame = self.frame_decoder(x_to_dec).squeeze() # [c,h,w]
        audio = self.audio_decoder(a_to_dec).squeeze()[-self.audio_f:] # [735,2]

        return frame, audio



            




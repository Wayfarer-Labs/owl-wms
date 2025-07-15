import torch

from owl_wms.models import get_model_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only
from owl_wms.configs import Config
from owl_wms.data import get_loader

from webapp.utils.configs import StreamingConfig

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

class ICMLPipeline:
    def __init__(self,
                 stream_config: StreamingConfig,
                 cfg_path="webapp/checkpoints/configs/icml.yml",
                 ckpt_path="webapp/checkpoints/models/causvid_ema.pt",
                 device="cuda"):
        self.device = device
        self.cfg = Config.from_yaml(cfg_path)
        self.model_cfg      = self.cfg.model
        self.train_cfg      = self.cfg.train
        self.stream_config  = stream_config
        
        self.model = get_model_cls(self.model_cfg.model_id)(self.model_cfg).core
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        self.frame_decoder = get_decoder_only(
            None,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        #self.audio_decoder = get_decoder_only(
        #    None,
        #    self.train_cfg.audio_vae_cfg_path,
        #    self.train_cfg.audio_vae_ckpt_path
        #)

        self.frame_scale = self.train_cfg.vae_scale
        self.audio_scale = self.train_cfg.audio_vae_scale

        self.history_buffer = None
        self.audio_buffer = None
        self.mouse_buffer = None
        self.button_buffer = None

        # loader = get_loader(
        #     "cod_s3_audio",
        #     1,
        #     window_length=self.model_cfg.n_frames,
        #     bucket_name='cod-data-latent-360x640to8x8'
        # )
        # self.loader = iter(loader)

        self.alpha = 0.2
        self.model = self.model.to(self.device)
        self.frame_decoder = self.frame_decoder.to(self.device)
        self.model = self.model.bfloat16()
        self.frame_decoder = self.frame_decoder.bfloat16()
        # self.model = torch.compile(self.model, mode = 'max-autotune', dynamic = False, fullgraph = True).to(self.device)
        # self.frame_decoder = torch.compile(self.frame_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True).to(self.device)
        #self.audio_decoder = torch.compile(audio_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True)

        self.audio_f = 735
        self.init_buffers_2()

    def init_buffers_2(self):
        self.history_buffer = torch.load(self.stream_config.video_latent_history_path, map_location=self.device)
        self.audio_buffer   = torch.load(self.stream_config.audio_latent_history_path, map_location=self.device)
        self.mouse_buffer   = torch.load(self.stream_config.mouse_history_path, map_location=self.device)
        self.button_buffer  = torch.load(self.stream_config.button_history_path, map_location=self.device)

    def init_buffers(self):
        self.history_buffer,self.audio_buffer,self.mouse_buffer,self.button_buffer=next(self.loader)
        # [1,n,c,h,w], [1,n,c], [1,n,2], [1,n,11]

        self.history_buffer = self.history_buffer.to(self.device) / self.frame_scale
        self.audio_buffer = self.audio_buffer.to(self.device) / self.audio_scale
        self.mouse_buffer = self.mouse_buffer.to(self.device)
        self.button_buffer = self.button_buffer.to(self.device)

    @torch.no_grad()
    def __call__(self, new_mouse, new_btn):
        # [2,] float and [11,] bool
        noised_history = zlerp(self.history_buffer[:,1:], self.alpha).to(self.device)
        noised_audio = zlerp(self.audio_buffer[:,1:], self.alpha).to(self.device)

        noised_history = torch.cat([noised_history, torch.randn_like(noised_history[:,0:1])], dim = 1)
        noised_audio = torch.cat([noised_audio, torch.randn_like(noised_audio[:,0:1])], dim = 1)

        new_mouse = new_mouse[None,None,:].to(self.device)
        new_btn = new_btn[None,None,:].to(self.device)

        self.mouse_buffer = torch.cat([self.mouse_buffer[:,1:],new_mouse],dim=1).to(self.device)
        self.button_buffer = torch.cat([self.button_buffer[:,1:],new_btn],dim=1).to(self.device)

        x = noised_history.to(self.device)
        a = noised_audio.to(self.device)
        ts = torch.ones_like(noised_history[:,:,0,0,0]).to(self.device)
        ts[:,:-1] = self.alpha

        prev_vid_batch, pred_aud_batch = self.model(x, a, ts, self.mouse_buffer, self.button_buffer)
        x[:,-1] = x[:,-1] - prev_vid_batch[:,-1].to(self.device)
        a[:,-1] = a[:,-1] - pred_aud_batch[:,-1].to(self.device)
        
        new_frame = x[:,-1:] # [1,1,c,h,w]
        new_audio = a[:,-1:] # [1,1,c]

        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1).to(self.device)
        self.audio_buffer = torch.cat([self.audio_buffer[:,1:], new_audio], dim=1).to(self.device)

        x_to_dec = new_frame[0] * self.frame_scale
        a_to_dec = self.audio_buffer * self.audio_scale

        frame = self.frame_decoder(x_to_dec).squeeze().to(self.device) # [c,h,w]
        #audio = self.audio_decoder(a_to_dec).squeeze()[-self.audio_f:] # [735,2]

        return frame, None
    


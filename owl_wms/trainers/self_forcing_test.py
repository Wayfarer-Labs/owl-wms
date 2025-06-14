import torch, random, math
from typing import Sequence, Optional, Callable, List
from torch.utils.data import DataLoader                # 📦 your own dataset

# toy helpers ---------------------------------------------------------------
def alpha(t):            # signal coefficient
    return torch.cos(torch.tensor(t, device='cuda') * torch.pi / 2 / 1000)
def sigma(t):            # noise coefficient
    return torch.sin(torch.tensor(t, device='cuda') * torch.pi / 2 / 1000)

def q_sample(x0, t):
    eps = torch.randn_like(x0)
    return alpha(t) * x0 + sigma(t) * eps, eps


@torch.no_grad()
def _pick_t(t_schedule: Sequence[int]) -> int:
    """Uniformly pick a timestep from the paper’s schedule."""
    return random.choice(tuple(t_schedule))


def compute_dmd_loss(
    *,
    fake_frames:   List[torch.Tensor],        # list of x̂_i_0 from your rollout
    real_clip:     torch.Tensor,              # batch of real clips, latent-space
    model:         torch.nn.Module,           # causal generator G_θ
    teacher:       torch.nn.Module,           # frozen bidirectional net
    q_sample_fn:      Callable[[torch.Tensor,int],  # function that adds noise
                           tuple[torch.Tensor, torch.Tensor]],
    t_schedule:    Sequence[int],
    score_fn:    Optional[Callable[[torch.Tensor,int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Return the Distribution-Matching Distillation (DMD) loss.
    
    Args
    ----
    fake_frames : list of N tensors shaped (C,F,H,W) – the generator’s roll-out  
    real_clip   : tensor shaped (N,C,F,H,W) from the dataset (already latent)  
    model       : generator; must expose `score_fn` if not passed explicitly  
    teacher     : frozen bidirectional model giving ∇log p_data  
    q_sample    : function that maps (clean, t) → (noisy, eps)  
    t_schedule  : iterable of the allowed timesteps (e.g. [1000,750,500,250])  
    score_fn  : optional callable; defaults to model.score_fn  
    """
    device = real_clip.device
    fake_video = torch.stack(fake_frames, dim=1)   # (B,N,C,F,H,W)

    # 1) pick a timestep and add the *same* noise level to real & fake
    t             = _pick_t(t_schedule)
    noisy_fake, _ = q_sample_fn(fake_video, t)
    noisy_real, _ = q_sample_fn(real_clip,  t)

    # 2) scores
    with torch.no_grad():                  # teacher is frozen
        s_real = teacher(noisy_real, t)

    if score_fn is None:
        # assume model implements the score head
        s_fake = model.score_fn(noisy_fake, t)
    else:
        s_fake = score_fn(noisy_fake, t)

    # 3) DMD loss  (Eq. 3 in the Self-Forcing paper)
    loss = 0.5 * ((s_fake - s_real.detach()) ** 2).mean()
    return loss


# --------------------------------------------------------------------------- #
# 0.  CONFIG – tweak here only                                                #
# --------------------------------------------------------------------------- #
device          = 'cuda'
batch_size      = 4             #   B
num_frames      = 60            #   N
t_schedule      = [1000, 750, 500, 250]
KV_max_size     = 60            # window L
lr              = 2e-6
total_steps     = 10_000
latent_shape    = (128, 4, 4)   # latents for an image
# --------------------------------------------------------------------------- #
# 1.  MODEL / TEACHER / OPTIMISER                                             #
# --------------------------------------------------------------------------- #

class G_theta(torch.nn.Module):
    ...

class Wan2_1_T2V_14B(torch.nn.Module):
    ...

model   = G_theta().to(device).train()               # causal generator
teacher = Wan2_1_T2V_14B.load_pretrained().eval().to(device)
teacher.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0, 0.999))

# --------------------------------------------------------------------------- #
# 2.  DATALOADER  ––  your dataset must yield latent videos of shape          #
#                    (B,N,C,F,H,W).  Here we fake one for demo.               #
# --------------------------------------------------------------------------- #
class DummySet(torch.utils.data.Dataset):
    def __len__(self):  return 10_000
    def __getitem__(self, idx):
        # latent video, random values just for code-run sanity
        vid = torch.randn(num_frames, *latent_shape)
        return vid
train_loader = DataLoader(DummySet(), batch_size, shuffle=True, drop_last=True)
loader_it    = iter(train_loader)

# --------------------------------------------------------------------------- #
# 3.  AUTOREGRESSIVE ROLLOUT  (wrapped into a helper)                         #
# --------------------------------------------------------------------------- #
def autoreg_rollout(model, B):
    X_theta: list[torch.Tensor] = []   # length N, each (B,C,F,H,W)
    KV: list[torch.Tensor]      = []   # rolling cache
    s = random.randint(1, len(t_schedule))          # gradient step to keep

    for i in range(num_frames):                        # ------ frame loop
        x_i_t = torch.randn(B, *latent_shape, device=device)   # noise at t=T

        for step_idx, t in enumerate(reversed(t_schedule), 1): # timestep loop
            keep_grads = (step_idx == s)

            with torch.set_grad_enabled(keep_grads):
                x_hat_i_0 = model(x_i_t, t=t, kv_cache=KV)

            if keep_grads:                                # keep grads on step s
                X_theta.append(x_hat_i_0)
                kv_i = model.get_kv(x_hat_i_0, t=0)
                if len(KV) == KV_max_size: KV.pop(0)      # rolling window
                KV.append(kv_i)
            else:
                x_hat_i_0 = x_hat_i_0.detach()

            if step_idx < len(t_schedule):                 # re-noisify
                eps     = torch.randn_like(x_hat_i_0)
                prev_t  = t_schedule[-(step_idx+1)]
                x_i_t   = alpha(prev_t)*x_hat_i_0 + sigma(prev_t)*eps

    return X_theta                                        # list length N
# --------------------------------------------------------------------------- #
# 4.  TRAIN LOOP                                                              #
# --------------------------------------------------------------------------- #
step = 0
while step < total_steps:
    # ------------------ 4.1  FETCH REAL CLIP --------------------------------
    try:
        real_clip = next(loader_it)
    except StopIteration:
        loader_it  = iter(train_loader)
        real_clip  = next(loader_it)
    real_clip = real_clip.to(device)                      # (B,N,C,F,H,W)
    B = real_clip.shape[0]

    # ------------------ 4.2  SELF-ROLLOUT -----------------------------------
    fake_frames = autoreg_rollout(model, B)               # list of (B,C,F,H,W)

    # ------------------ 4.3  DMD LOSS  +  BACKWARD --------------------------
    loss = compute_dmd_loss(
        fake_frames = fake_frames,
        real_clip   = real_clip,
        model       = model,
        teacher     = teacher,
        q_sample_fn = q_sample,
        t_schedule  = t_schedule
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()                                       # grads only through s
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ------------------ 4.4  LOG ------------------------------------------------
    if step % 100 == 0:
        print(f"[{step:05d}]  loss = {loss.item():.4f}")

    step += 1

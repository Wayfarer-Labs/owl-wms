import torch
import torch.nn.functional as F

from .craft_trainer import CraftTrainer


class RFTPairDistillTrainer(CraftTrainer):
    def fwd_step(self, batch, train_step: int):
        return self.ayf_emd(batch)
        """
        if train_step < self.train_cfg.finite_difference_step:
            return self.ode_fwd(batch)
        else:
            return self.flow_matching_fwd(batch)
        """

    def ode_fwd(self, batch):
        x_a, t_a, _, _, x_clean, t_clean = batch
        with self.autocast_ctx:
            pred_x0 = self.core_fwd(x_a, t_a)
        return F.mse_loss(pred_x0.float(), x_clean.float())

    def flow_matching_fwd(self, batch, u_frac=0.0, noise_std=0.0):
        x_a, t_a, x_b, t_b, _, _ = batch
        assert t_a.dtype == torch.float32

        # sample interpolated point
        if u_frac is None:
            u_frac = torch.rand_like(t_a)
        else:
            u_frac = torch.full_like(t_a, u_frac)

        # inputs
        lam = u_frac.reshape(*u_frac.shape, *([1] * (x_a.dim() - u_frac.dim())))
        x_u = (1 - lam) * x_a + lam * x_b
        if noise_std:
            x_u = x_u + noise_std * torch.randn_like(x_u)

        s_u = (1 - u_frac) * t_a + u_frac * t_b

        # derivative
        dt = (t_b - t_a)  # shape [B]
        assert not torch.any(dt == 0)
        dt = dt.reshape(*dt.shape, *([1] * (x_a.dim() - dt.dim())))
        v_hat = (x_b - x_a) / dt

        """
        # 3) model forward + loss
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)
        return F.mse_loss(v_pred.float(), v_hat.float())  # TODO: scale by dt
        """
        with self.autocast_ctx:
            v_pred = self.core_fwd(x_u, s_u)
        # per-example MSE, then weight by |dt| (approximate ∫ over time)
        mse = F.mse_loss(v_pred.float(), v_hat.float(), reduction='none')
        per_ex = mse.reshape(mse.size(0), -1).mean(dim=1)      # [B]
        w = (dt ** 2) / (dt ** 2).mean()               # normalize weights
        return (w * per_ex).mean()

    def ayf_emd_old(
        self,
        batch,
        tangent_norm: bool = True,
        local_span: float = 0.05,
    ):
        """
        offline AYF-EMD (one-step Euler):
          s=(x_a,t_a)  u=(x_b,t_b) with t_a > t_b
          Compare: direct s→t  vs   two-step s→u→t (via-u under stop-grad)
          t is chosen ≤ u and ≤ s (forward in time for both paths)
          No phi, t in [0, 1], no warping
        """
        x_a, t_a, x_b, t_b = batch[:4]

        # ----- choose target time in RAW clock (not φ) -----
        # cap span by how much room remains to t=0
        # sample a strictly positive local step below u (since t_a > t_b)
        rho = torch.rand_like(t_b).clamp_min(1e-2)  # avoid ~0 step
        span_cap = torch.minimum(torch.full_like(t_b, local_span), t_b)
        t_raw = (t_b - rho * span_cap).clamp_min(0.0)  # strictly < t_b

        def expand_like(ts: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return ts.view(ts.shape + (1,) * (ref.ndim - ts.ndim))

        # RAW Δt for Euler steps; broadcast to match x_*
        dt_s = expand_like(t_raw - t_a, x_a).to(x_a.dtype)  # Δt from s to t (≤ 0)
        dt_u = expand_like(t_raw - t_b, x_b).to(x_b.dtype)  # Δt from u to t (≤ 0)

        # Optional per-sample tangent normalization
        def normalize_tangent(v: torch.Tensor) -> torch.Tensor:
            denom = v.pow(2).mean(dim=tuple(range(1, v.ndim)), keepdim=True).sqrt().clamp_min(1e-6)
            return v / denom.detach()  # supervise direction; stop scale-grad

        # ----- direct branch: s → t (grads on) -----
        with self.autocast_ctx:
            v_a = self.core_fwd(x_a, t_a)
            if tangent_norm:
                v_a = normalize_tangent(v_a)
        # do the Euler math in fp32 for stability, then cast back
        x_direct = (x_a.float() + dt_s.float() * v_a.float()).to(x_a.dtype)

        # ----- via-u branch: u → t (STOP-GRAD) -----
        with torch.no_grad():
            with self.autocast_ctx:
                v_b = self.core_fwd(x_b, t_b)
                if tangent_norm:
                    v_b = normalize_tangent(v_b)
            x_via_u = (x_b.float() + dt_u.float() * v_b.float()).to(x_b.dtype)

        return F.mse_loss(x_direct.float(), x_via_u.float())

    def ayf_emd(
        self,
        batch,
        tangent_norm: bool = True,
        local_span: float = 0.05,
        step: int = 0,
        anchor_lambda0: float = 0.10,
        anchor_decay_steps: int = 2000,
        tn_warmup_steps: int = 4000,
    ):
        """
        Offline AYF-EMD https://arxiv.org/pdf/2506.14603
          one-step Euler
          raw clock t in [0,1], no warping.
          s=(x_a,t_a), u=(x_b,t_b), with t_a > t_b.
          Compare: direct s→t (grads on) vs via-u u→t (stop-grad).
          Includes: gap-tied target, dt hygiene, per-example reduction, clipped step-size weighting,
                    tiny anchor with cosine decay to 0, and tangent-norm warmup.
        """
        x_a, t_a, x_b, t_b, x_next, t_next = batch[:6]

        # ----- choose target time (gap-tied near u, with small exploration) -----
        delta = (t_a - t_b).abs()
        alpha = torch.empty_like(t_b).uniform_(0.3, 0.9)
        span_cap = torch.minimum(torch.full_like(t_b, local_span), 0.75 * delta)
        t_gap_tied = (t_b - alpha * span_cap).clamp_min(0.0)
        explore_mask = (torch.rand_like(t_b) < 0.12)
        t_uni = torch.rand_like(t_b) * t_b
        t_raw = torch.where(explore_mask, t_uni, t_gap_tied)
        # strict t < t_b (and < t_a), then keep within [0,1]
        _eps = 1e-6
        t_raw = torch.minimum(t_raw, t_b - _eps)
        t_raw = torch.minimum(t_raw, t_a - _eps).clamp_min(0.0)

        def expand_like(ts: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return ts.view(ts.shape + (1,) * (ref.ndim - ts.ndim))

        # RAW Δt for Euler steps; broadcast to match x_*
        dt_s = expand_like(t_raw - t_a, x_a).to(x_a.dtype)  # Δt from s to t (≤ 0)
        dt_u = expand_like(t_raw - t_b, x_b).to(x_b.dtype)  # Δt from u to t (≤ 0)

        # Optional per-sample tangent normalization
        def normalize_tangent(v: torch.Tensor) -> torch.Tensor:
            assert v.dtype == torch.float32
            denom = v.pow(2).mean(dim=tuple(range(1, v.ndim)), keepdim=True).sqrt().clamp_min(1e-6)
            return v / denom.detach()  # supervise direction; stop scale-grad

        # ----- direct branch: s → t (grads on) -----
        with self.autocast_ctx:
            v_a_raw = self.core_fwd(x_a, t_a)

        # ---- AYF Tangent Normalization (TN) with warmup (paper §3.4) ----
        # Warmup coefficient linearly increases, clamped < 1 to keep some regularization.
        tn_coeff = 0.0
        if tangent_norm:
            if tn_warmup_steps > 0:
                tn_coeff = min(0.9, float(step) / float(tn_warmup_steps))
            else:
                tn_coeff = 0.9

        def blend_tn(v: torch.Tensor) -> torch.Tensor:
            # Per-sample L2 normalization over non-batch dims; detach scale per sCM/AYF intent.
            v32 = v.float()
            denom = v32.pow(2).mean(dim=tuple(range(1, v32.ndim)), keepdim=True).sqrt().clamp_min(1e-6)
            v_norm = v32 / denom.detach()
            # Interpolate between raw and normalized tangent (TN on) per warmup.
            return ((1.0 - tn_coeff) * v32 + tn_coeff * v_norm).to(v.dtype)

        v_a_eff = blend_tn(v_a_raw) if tangent_norm else v_a_raw
        # Euler update with TN-applied tangent
        x_direct = (x_a.float() + dt_s.float() * v_a_eff.float()).to(x_a.dtype)

        # ----- via-u branch: student(u → t) with STOP-GRAD (AYF-EMD) -----
        # Use the recorded adjacent teacher state as x^u := x_b.
        with torch.no_grad():  # stop-grad on the via-u student map
            v_b = self.core_fwd(x_b, t_b)          # student velocity at u
            v_b_eff = blend_tn(v_b) if tangent_norm else v_b  # apply SAME TN policy
            x_via_u = (x_b.float() + dt_u.float() * v_b_eff.float()).to(x_b.dtype)

        # ----- per-example loss with clipped step-size weighting -----
        diff = x_direct.float() - x_via_u.float()
        per_ex = diff.pow(2).mean(dim=tuple(range(1, diff.ndim)))  # per-sample MSE

        # Paper recommendation: constant weight
        main_loss = per_ex.mean()
        return main_loss

    @torch.compile
    def core_fwd(self, *args, **kwargs):
        core = self.get_module(ema=False).core
        return core(*args, **kwargs)

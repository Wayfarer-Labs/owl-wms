from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
import warnings


def init_muon(model, rank: int = 0, world_size: int = 1, **kwargs):
    """
    Build optimizer param groups. Only hyperparameters you pass override muon.py defaults
    anything omitted is left out so the optimizer applies its own defaults.
    """
    adamw_keys = set(kwargs.get("adamw_keys", []))
    low_lr_keys = set(kwargs.get("adamw_low_lr_keys", []))
    low_lr_mul = kwargs.get("adamw_low_lr_mul", 0.1)

    # normalize names like before
    named = {n.replace("._orig_mod", ""): p for n, p in model.named_parameters()}

    # validate keys
    names = list(named.keys())
    for key in adamw_keys:
        if not any(key in n for n in names):
            warnings.warn(f"AdamW key '{key}' not found in model parameters")

    # split
    match = lambda name, keys: any(k in name for k in keys)
    adam_base = [p for n, p in named.items() if not match(n, low_lr_keys) and (match(n, adamw_keys) or p.ndim < 2)]
    adam_low_lr = [p for n, p in named.items() if match(n, low_lr_keys)]
    muon_params = [p for n, p in named.items() if not match(n, low_lr_keys) and not match(n, adamw_keys) and p.ndim >= 2]

    # only include overrides that are not None
    adam_overrides = {
        "lr": kwargs.get("adamw_lr"),
        "betas": kwargs.get("adamw_betas"),
        "weight_decay": kwargs.get("adamw_wd"),
        "eps": kwargs.get("adamw_eps"),
        "use_muon": False,
    }
    adam_overrides = {k: v for k, v in adam_overrides.items() if v is not None}

    muon_overrides = {
        "lr": kwargs.get("lr"),
        "momentum": kwargs.get("momentum"),
        "weight_decay": kwargs.get("weight_decay"),
        "use_muon": True,
    }
    muon_overrides = {k: v for k, v in muon_overrides.items() if v is not None}

    adam_base_group = {"params": adam_base, **adam_overrides}
    adam_low_lr_group = {
        "params": adam_low_lr,
        **{k: v for k, v in adam_overrides.items() if k != "lr"},
        **({"lr": adam_overrides["lr"] * float(low_lr_mul)} if "lr" in adam_overrides else {}),
    }

    groups = [
        {**adam_base_group, "params": [p for p in adam_base if p.ndim >= 2]},
        {**adam_base_group, "params": [p for p in adam_base if p.ndim < 2], "weight_decay": 0.0},
        {**adam_low_lr_group, "params": [p for p in adam_low_lr if p.ndim >= 2]},
        {**adam_low_lr_group, "params": [p for p in adam_low_lr if p.ndim < 2], "weight_decay": 0.0},
        {"params": muon_params, **muon_overrides}
    ]

    OptimizerCls = SingleDeviceMuonWithAuxAdam if world_size == 1 else MuonWithAuxAdam
    return OptimizerCls([g for g in groups if g["params"]])

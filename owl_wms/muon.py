from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


def init_muon(model, rank: int = 0, world_size: int = 1, **kwargs):
    """
    Build param groups from `adamw_keys` (and ndim). Only hyperparameters you pass
    override muon.py defaults; anything omitted is left out of the group so the
    optimizer applies its own defaults.
    """
    adamw_keys = kwargs.get("adamw_keys", [])

    # normalize names like before
    named = {n.replace("._orig_mod", ""): p for n, p in model.named_parameters()}

    # validate keys
    names = list(named.keys())
    for key in adamw_keys:
        assert any(key in n for n in names), (
            f"AdamW key '{key}' not found in model parameters: {names}"
        )

    # split
    adam_params = [p for n, p in named.items() if any(k in n for k in adamw_keys) or p.ndim < 2]
    muon_params = [p for n, p in named.items() if not any(k in n for k in adamw_keys) and p.ndim >= 2]

    # only include overrides that are not None
    adam_overrides = {
        "lr": kwargs.get("adamw_lr"),
        "betas": kwargs.get("adamw_betas"),
        "eps": kwargs.get("adamw_eps"),
        "weight_decay": kwargs.get("adamw_wd"),
    }
    adam_group = {
        "params": adam_params,
        "use_muon": False,
        **{k: v for k, v in adam_overrides.items() if v is not None},
    }

    muon_overrides = {
        "lr": kwargs.get("lr"),
        "momentum": kwargs.get("momentum"),
        "weight_decay": kwargs.get("weight_decay"),
    }
    muon_group = {
        "params": muon_params,
        "use_muon": True,
        **{k: v for k, v in muon_overrides.items() if v is not None},
    }

    OptimizerCls = SingleDeviceMuonWithAuxAdam if world_size == 1 else MuonWithAuxAdam
    return OptimizerCls([adam_group, muon_group])

def get_loader(data_id, batch_size, **data_kwargs):
    if data_id == "multi":
        from . import multi
        return multi.get_loader(batch_size, **data_kwargs)
    elif data_id == "wan_pair_distill":
        from . import wan_distillation_loader
        return wan_distillation_loader.get_pair_loader(batch_size, **data_kwargs)
    elif data_id == "wan_sample":
        from . import wan_distillation_loader
        return wan_distillation_loader.get_sample_loader(batch_size, **data_kwargs)
    elif data_id == "sequence_packing":
        from . import latent_seq_packing
        return latent_seq_packing.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod":
        from . import cod_latent
        return cod_latent.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3":
        from . import s3_cod_latent
        return s3_cod_latent.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3_audio":
        from . import s3_cod_latent_audio
        return s3_cod_latent_audio.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3_mixed":
        from . import s3_cod_latent_mixed
        return s3_cod_latent_mixed.get_loader(batch_size, **data_kwargs)

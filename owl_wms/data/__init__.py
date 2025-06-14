from . import local_cod_data
from . import local_cod_latent

def get_loader(data_id, batch_size, **data_kwargs):
    if data_id == "cod_data":
        return local_cod_data.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_latent":
        return local_cod_latent.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3":
        from . import s3_cod_latent
        return s3_cod_latent.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3_audio":
        from . import s3_cod_latent_audio
        return s3_cod_latent_audio.get_loader(batch_size, **data_kwargs)
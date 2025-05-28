from . import local_cod_data
from . import local_cod_latent
def get_loader(data_id, batch_size):
    if data_id == "cod_data":
        return local_cod_data.get_loader(batch_size)
    elif data_id == "cod_latent":
        return local_cod_latent.get_loader(batch_size)
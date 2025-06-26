import numpy as np

def random_reward(*args):
    # assumes first arg has a batch size...
    batch_size = len(args[0])

    return np.random.random(batch_size)
import boto3
import threading
from dotenv import load_dotenv
import os

load_dotenv()

import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import tarfile
import io
import time

class RandomizedQueue:
    def __init__(self):
        self.items = []

    def add(self, item):
        idx = random.randint(0, len(self.items))
        self.items.insert(idx, item)

    def pop(self):
        if not self.items:
            return None
        idx = random.randint(0, len(self.items) - 1)
        return self.items.pop(idx)

TOTAL_SHARDS = 2
NUM_SUBDIRS=1
NUM_TARS=9
BUCKET_NAME="cod-data-latent-360x640to5x8"

class S3CoDLatentDataset(IterableDataset):
    def __init__(self, window_length=120, file_share_max=20, rank=0, world_size=1, bucket_name = BUCKET_NAME, include_keyframe = False):
        super().__init__()
        
        self.window = window_length
        self.file_share_max = file_share_max
        self.rank = rank
        self.world_size = world_size
        self.include_keyframe = include_keyframe
        self.bucket_name = bucket_name

        # Queue parameters
        self.max_tars = 2
        self.max_data = 1000

        # Initialize queues
        self.tar_queue = RandomizedQueue()
        self.data_queue = RandomizedQueue()

        # Setup S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
        )

        # Start background threads
        self.tar_thread = threading.Thread(target=self.background_download_tars, daemon=True)
        self.data_thread = threading.Thread(target=self.background_load_data, daemon=True)
        self.tar_thread.start()
        self.data_thread.start()

    def random_sample_prefix(self):
        # For now just 2 shards (00, 01)
        shard = random.randint(0, TOTAL_SHARDS-1)
        # Each shard has 1000 subdirs
        subdir = random.randint(0, NUM_SUBDIRS-1)
        # Each subdir has multiple tars
        tar_num = random.randint(0, NUM_TARS-1)
        return f"{shard:02d}/{subdir:04d}/{tar_num:04d}.tar"

    def background_download_tars(self):
        while True:
            if len(self.tar_queue.items) < self.max_tars:
                tar_path = self.random_sample_prefix()
                try:
                    # Download tar directly to memory
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=tar_path)
                    tar_data = response['Body'].read()
                    self.tar_queue.add(tar_data)
                except Exception as e:
                    print(f"Error downloading tar {tar_path}: {e}")
            else:
                time.sleep(1)

    def process_tensor_file(self, tar, base_name, suffix):
        try:
            f = tar.extractfile(f"{base_name}.{suffix}.pt")
            if f is not None:
                tensor_data = f.read()
                tensor = torch.load(io.BytesIO(tensor_data))
                return tensor
        except:
            return None
        return None

    def background_load_data(self):
        while True:
            if len(self.data_queue.items) < self.max_data:
                tar_data = self.tar_queue.pop()
                if tar_data is None:
                    time.sleep(1)
                    continue

                try:
                    tar_file = io.BytesIO(tar_data)
                    with tarfile.open(fileobj=tar_file) as tar:
                        members = tar.getmembers()
                        base_names = set()
                        
                        # Get unique base names
                        for member in members:
                            if member.name.endswith('.latent.pt'):
                                base_names.add(member.name.split('.')[0])

                        for base_name in base_names:
                            # Load all tensors for this base name
                            latent = self.process_tensor_file(tar, base_name, "latent")
                            mouse = self.process_tensor_file(tar, base_name, "mouse")
                            button = self.process_tensor_file(tar, base_name, "buttons")

                            if all(t is not None for t in [latent, mouse, button]):
                                min_len = min(len(latent), len(mouse), len(button))
                                
                                # Sample multiple windows if requested
                                for _ in range(self.file_share_max):
                                    if len(self.data_queue.items) >= self.max_data:
                                        break
                                        
                                    max_start = min_len - self.window
                                    if max_start <= 0:
                                        continue
                                        
                                    window_start = random.randint(0, max_start)
                                    
                                    latent_slice = latent[window_start:window_start+self.window].float()
                                    mouse_slice = mouse[window_start:window_start+self.window]
                                    button_slice = button[window_start:window_start+self.window]

                                    if self.include_keyframe:
                                        # Sample keyframe from nearby in video but not in window
                                        buffer = 400
                                        valid_range_start = max(0, window_start - buffer)
                                        valid_range_end = min(len(latent), window_start + self.window + buffer)
                                        
                                        # Exclude the actual window frames
                                        valid_frames = list(range(valid_range_start, window_start)) + \
                                                     list(range(window_start + self.window, valid_range_end))
                                        
                                        if valid_frames:
                                            keyframe_idx = random.choice(valid_frames)
                                            latent_keyframe = latent[keyframe_idx].float().unsqueeze(0)
                                            self.data_queue.add((latent_slice, latent_keyframe, mouse_slice, button_slice))
                                    else:
                                        self.data_queue.add((latent_slice, mouse_slice, button_slice))

                except Exception as e:
                    print(f"Error processing tar: {e}")
            else:
                time.sleep(1)

    def __iter__(self):
        while True:
            item = self.data_queue.pop()
            if item is not None:
                yield item
            else:
                time.sleep(0.1)

def collate_fn(batch):
    # batch is list of triples or quads
    items = zip(*batch)
    items = list(items)
    
    if len(items) == 3:
        # No keyframe case
        latents, mouses, buttons = items
        latents = torch.stack(latents)    # [b,n,c,h,w]
        mouses = torch.stack(mouses)      # [b,n,2] 
        buttons = torch.stack(buttons)    # [b,n,n_buttons]
        return latents, mouses, buttons
    else:
        # With keyframe case
        latents, keyframes, mouses, buttons = items
        latents = torch.stack(latents)      # [b,n,c,h,w]
        keyframes = torch.stack(keyframes)  # [b,1,c,h,w]
        mouses = torch.stack(mouses)        # [b,n,2]
        buttons = torch.stack(buttons)      # [b,n,n_buttons]
        return latents, keyframes, mouses, buttons

def get_loader(batch_size, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = S3CoDLatentDataset(rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    import time
    loader = get_loader(256, window_length = 120, file_share_max = 20)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    first_time = end - start
    
    start = time.time()
    batch = next(iter(loader)) 
    end = time.time()
    second_time = end - start
    
    x,y,z = batch
    print(f"Time to load first batch: {first_time:.2f}s")
    print(f"Time to load second batch: {second_time:.2f}s")
    print(f"Video shape: {x.shape}")
    print(x.std())
    print(f"Mouse shape: {y.shape}") 
    print(f"Button shape: {z.shape}")

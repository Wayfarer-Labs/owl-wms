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

class S3CoDLatentAudioDataset(IterableDataset):
    def __init__(self, window_length=120, file_share_max=20, rank=0, world_size=1, 
        bucket_name = "cod-latent-depth-4x4",
        cond_prefix="labelled",
        uncond_prefix="unlabelled",
        unlabelled_frac=0.5,
        verbose=False
    ):
        super().__init__()
        
        self.bucket_name = bucket_name
        self.window = window_length
        self.file_share_max = file_share_max
        self.rank = rank
        self.world_size = world_size
        self.cond_prefix = cond_prefix
        self.uncond_prefix = uncond_prefix
        self.unlabelled_frac = unlabelled_frac
        self.verbose = verbose

        # Queue parameters
        self.max_tars = 2
        self.max_data = 1000

        # Initialize separate queues for conditional and unconditional data
        self.cond_tar_queue = RandomizedQueue()
        self.cond_data_queue = RandomizedQueue()
        
        # Only initialize uncond queues if we're using them
        if unlabelled_frac > 0:
            self.uncond_tar_queue = RandomizedQueue()
            self.uncond_data_queue = RandomizedQueue()

        # Setup S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
        )

        # Get list of available tars
        self.cond_tars = self.list_tars(self.cond_prefix)
        if unlabelled_frac > 0:
            self.uncond_tars = self.list_tars(self.uncond_prefix)

        # Start background threads
        self.cond_tar_thread = threading.Thread(target=lambda: self.background_download_tars(True), daemon=True)
        self.cond_data_thread = threading.Thread(target=lambda: self.background_load_data(True), daemon=True)
        
        self.cond_tar_thread.start()
        self.cond_data_thread.start()

        # Only start uncond threads if we're using them
        if unlabelled_frac > 0:
            self.uncond_tar_thread = threading.Thread(target=lambda: self.background_download_tars(False), daemon=True)
            self.uncond_data_thread = threading.Thread(target=lambda: self.background_load_data(False), daemon=True)
            self.uncond_tar_thread.start()
            self.uncond_data_thread.start()

    def list_tars(self, prefix):
        tars = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.tar'):
                        tars.append(obj['Key'])
        return tars

    def background_download_tars(self, is_conditional):
        queue = self.cond_tar_queue if is_conditional else self.uncond_tar_queue
        prefix = self.cond_prefix if is_conditional else self.uncond_prefix
        tars = self.cond_tars if is_conditional else self.uncond_tars
        
        if self.verbose: print(f"Starting {'cond' if is_conditional else 'uncond'} tar download thread")
        
        while True:
            if len(queue.items) < self.max_tars:
                tar_path = random.choice(tars)
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=tar_path)
                    tar_data = response['Body'].read()
                    queue.add(tar_data)
                    if self.verbose: print(f"Added tar to {'cond' if is_conditional else 'uncond'} queue. Queue size: {len(queue.items)}")
                except Exception as e:
                    print(f"Error downloading tar {tar_path}: {e}")
            else:
                time.sleep(1)
        
    def sleep_until_queues_filled(self):
        while True:
            cond_tar_filled = len(self.cond_tar_queue.items) >= self.max_tars
            cond_data_filled = len(self.cond_data_queue.items) >= self.max_data
            
            if self.unlabelled_frac > 0:
                uncond_tar_filled = len(self.uncond_tar_queue.items) >= self.max_tars
                uncond_data_filled = len(self.uncond_data_queue.items) >= self.max_data
                all_filled = all([cond_tar_filled, uncond_tar_filled, cond_data_filled, uncond_data_filled])
            else:
                all_filled = all([cond_tar_filled, cond_data_filled])
                
            if all_filled:
                break
                
            time.sleep(1)
            
            if self.verbose:
                status = f"Waiting for queues to fill... Tar queues: {len(self.cond_tar_queue.items)}/{self.max_tars}, "
                if self.unlabelled_frac > 0:
                    status += f"{len(self.uncond_tar_queue.items)}/{self.max_tars}, "
                status += f"Data queues: {len(self.cond_data_queue.items)}/{self.max_data}"
                if self.unlabelled_frac > 0:
                    status += f", {len(self.uncond_data_queue.items)}/{self.max_data}"
                print(status)

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

    def background_load_data(self, is_conditional):
        tar_queue = self.cond_tar_queue if is_conditional else self.uncond_tar_queue
        data_queue = self.cond_data_queue if is_conditional else self.uncond_data_queue
        prefix = "conditional" if is_conditional else "unconditional"
        
        while True:
            if len(data_queue.items) < self.max_data:
                tar_data = tar_queue.pop()
                if tar_data is None:
                    time.sleep(1)
                    continue

                try:
                    tar_file = io.BytesIO(tar_data)
                    with tarfile.open(fileobj=tar_file) as tar:
                        members = tar.getmembers()
                        base_names = set()
                        
                        for member in members:
                            if member.name.endswith('.latent.pt'):
                                base_names.add(member.name.split('.')[0])

                        for base_name in base_names:
                            latent = self.process_tensor_file(tar, base_name, "latent")
                            latent = torch.clamp(latent, -8, 8)
                            latent = torch.nan_to_num(latent, nan=0.0)
                            audio = self.process_tensor_file(tar, base_name, "audiolatent")
                            
                            if is_conditional:
                                mouse = self.process_tensor_file(tar, base_name, "mouse")
                                button = self.process_tensor_file(tar, base_name, "buttons")
                            else:
                                # Create zero tensors for unconditional data
                                if latent is not None:
                                    mouse = torch.zeros(len(latent), 2)
                                    button = torch.zeros(len(latent), 11)  # Assuming 11 buttons
                                else:
                                    mouse = button = None

                            if all(t is not None for t in [latent, mouse, button, audio]):
                                min_len = min(len(latent), len(mouse), len(button), len(audio))
                                
                                for _ in range(self.file_share_max):
                                    if len(data_queue.items) >= self.max_data:
                                        break
                                        
                                    max_start = min_len - self.window
                                    if max_start <= 0:
                                        continue
                                        
                                    window_start = random.randint(0, max_start)
                                    
                                    latent_slice = latent[window_start:window_start+self.window].float()
                                    mouse_slice = mouse[window_start:window_start+self.window]
                                    button_slice = button[window_start:window_start+self.window]
                                    audio_slice = audio[window_start:window_start+self.window]

                                    data_queue.add((latent_slice, mouse_slice, button_slice, audio_slice, is_conditional))

                except Exception as e:
                    print(f"Error processing tar: {e}")
            else:
                time.sleep(1)

    def __iter__(self):
        while True:
            # If unlabelled_frac is 0, only use conditional data
            if self.unlabelled_frac == 0:
                item = self.cond_data_queue.pop()
                if item is None and self.verbose:
                    print(f"Cond queue empty! Queue size: {len(self.cond_data_queue.items)}")
            else:
                # Sample based on unlabelled_frac
                should_be_uncond = random.random() < self.unlabelled_frac
                if should_be_uncond:
                    item = self.uncond_data_queue.pop()
                    if item is None and self.verbose:
                        print(f"Uncond queue empty! Queue sizes - Cond: {len(self.cond_data_queue.items)}, Uncond: {len(self.uncond_data_queue.items)}")
                else:
                    item = self.cond_data_queue.pop()
                    if item is None and self.verbose:
                        print(f"Cond queue empty! Queue sizes - Cond: {len(self.cond_data_queue.items)}, Uncond: {len(self.uncond_data_queue.items)}")
                    
            if item is not None:
                yield item
            else:
                time.sleep(0.1)

def collate_fn(batch):
    # batch is list of quintuples (latent, mouse, button, audio, is_conditional)
    latents, mouses, buttons, audios, has_controls = zip(*batch)
    
    latents = torch.stack(latents)      # [b,n,c,h,w]
    mouses = torch.stack(mouses)        # [b,n,2] 
    buttons = torch.stack(buttons).float()      # [b,n,n_buttons]
    audios = torch.stack(audios)        # [b,n,d]
    has_controls = torch.tensor(has_controls, dtype=torch.bool)  # [b,]
    
    return latents, audios, mouses, buttons, has_controls

def get_loader(batch_size, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = S3CoDLatentAudioDataset(rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/av_v5_8x8_mixed.yml")
    loader = get_loader(cfg.train.target_batch_size, **cfg.train.data_kwargs)
    import time

    loader.dataset.sleep_until_queues_filled()

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    first_time = end - start
    
    start = time.time()
    batch = next(iter(loader)) 
    end = time.time()
    second_time = end - start
    
    x, a, m, b, h = batch
    print(f"Time to load first batch: {first_time:.2f}s")
    print(f"Time to load second batch: {second_time:.2f}s")
    print(f"Video shape: {x.shape}")
    print(f"Audio shape: {a.shape}")
    print(f"Mouse shape: {m.shape}")
    print(f"Button shape: {b.shape}")
    print(f"Has controls: {h}")

    print(f"Proportion of conditional samples: {h.float().mean().item():.3f}")

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
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError, SSLError, ReadTimeoutError, ResponseStreamingError

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
    # Class-level semaphore to limit concurrent S3 requests across all instances
    # Use rank-based scaling: fewer concurrent requests per node
    _s3_semaphore = None  # Will be initialized per rank
    
    def __init__(self, window_length=120, file_share_max=20, rank=0, world_size=1, 
                 bucket_name="cod-latent-depth-4x4",
                 prefix="labelled", verbose = False):
        super().__init__()
        
        self.window = window_length
        self.file_share_max = file_share_max
        self.rank = rank
        self.world_size = world_size
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.verbose = verbose

        # Initialize rank-based semaphore (allow more concurrent requests)
        if S3CoDLatentAudioDataset._s3_semaphore is None:
            S3CoDLatentAudioDataset._s3_semaphore = threading.Semaphore(8)  # 8 concurrent requests across all ranks
        
        # Stagger initialization to avoid thundering herd
        stagger_delay = self.rank * 2.0 + random.uniform(0, 2.0)
        if verbose:
            print(f"[Rank {self.rank}] Staggering S3 init by {stagger_delay:.1f}s")
        time.sleep(stagger_delay)

        # Queue parameters
        self.max_tars = 2
        self.max_data = 1000

        # Initialize queues
        self.tar_queue = RandomizedQueue()
        self.data_queue = RandomizedQueue()

        # Setup S3 client with enhanced configuration for multi-node SSL
        config = Config(
            retries={'max_attempts': 10, 'mode': 'adaptive'},
            max_pool_connections=1,  # Single connection per rank
            connect_timeout=60,
            read_timeout=300,  # 5 minutes for very slow responses
            signature_version='s3v4',
            # SSL-specific optimizations
            parameter_validation=False,
            tcp_keepalive=True
        )
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
            config=config
        )

        # Get list of available tars
        self.tars = self.list_tars(self.prefix)

        # Start background threads
        self.tar_thread = threading.Thread(target=self.background_download_tars, daemon=True)
        self.data_thread = threading.Thread(target=self.background_load_data, daemon=True)
        self.tar_thread.start()
        self.data_thread.start()

    def list_tars(self, prefix):
        tars = []
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with self._s3_semaphore:
                    paginator = self.s3_client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                if obj['Key'].endswith('.tar'):
                                    tars.append(obj['Key'])
                    return tars
            except (ConnectionClosedError, SSLError, ClientError, ReadTimeoutError, ResponseStreamingError) as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Error listing tars (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    if self.verbose:
                        print(f"Failed to list tars after {max_retries} attempts: {e}")
                    raise
        return tars

    def background_download_tars(self):
        while True:
            if len(self.tar_queue.items) < self.max_tars:
                tar_path = random.choice(self.tars)
                max_retries = 3
                retry_delay = 1
                
                if self.verbose:
                    print(f"[Rank {self.rank}] Attempting to download tar: {tar_path}")
                
                for attempt in range(max_retries):
                    try:
                        with self._s3_semaphore:
                            if self.verbose:
                                print(f"[Rank {self.rank}] Starting download attempt {attempt + 1}")
                            
                            # Fast streaming with large chunks
                            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=tar_path)
                            if self.verbose:
                                print(f"[Rank {self.rank}] Got response, streaming data...")
                            
                            # Stream with large chunks for speed
                            tar_data = b''
                            content_length = response.get('ContentLength', 0)
                            
                            try:
                                for chunk in response['Body'].iter_chunks(chunk_size=50*1024*1024*3):  # 150MB chunks
                                    tar_data += chunk
                                    
                            except Exception as stream_e:
                                if self.verbose:
                                    print(f"[Rank {self.rank}] Streaming error: {stream_e}")
                                raise stream_e
                            
                            
                            self.tar_queue.add(tar_data)
                            if self.verbose:
                                print(f"[Rank {self.rank}] Added to queue successfully")
                            break  # Success, exit retry loop
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"[Rank {self.rank}] Error downloading tar {tar_path} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            if self.verbose:
                                print(f"[Rank {self.rank}] Failed to download tar {tar_path} after {max_retries} attempts, skipping")
                            break
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
                            latent = self.process_tensor_file(tar, base_name, "latent").bfloat16()
                            latent = torch.clamp(latent, -8, 8)
                            latent = torch.nan_to_num(latent, nan=0.0)
                            
                            mouse = self.process_tensor_file(tar, base_name, "mouse")
                            button = self.process_tensor_file(tar, base_name, "buttons")
                            
                            # If mouse/button don't exist, create zeros with correct shapes
                            if mouse is None or button is None:
                                n,_,_,_ = latent.shape
                                mouse = torch.zeros(n, 2)
                                button = torch.zeros(n, 11)
                            audio = self.process_tensor_file(tar, base_name, "audiolatent")

                            if all(t is not None for t in [latent, mouse, button, audio]):
                                min_len = min(len(latent), len(mouse), len(button), len(audio))
                                
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
                                    audio_slice = audio[window_start:window_start+self.window]

                                    self.data_queue.add((latent_slice, mouse_slice, button_slice, audio_slice))

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
    # batch is list of quadruples
    latents, mouses, buttons, audios = zip(*batch)
    
    latents = torch.stack(latents).bfloat16()    # [b,n,c,h,w]
    mouses = torch.stack(mouses).bfloat16()      # [b,n,2] 
    buttons = torch.stack(buttons).bfloat16()    # [b,n,n_buttons]
    audios = torch.stack(audios).bfloat16()      # [b,n,d]
    
    return latents, audios, mouses, buttons

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
    import time
    loader = get_loader(16, 
                       window_length=16,
                       bucket_name="cod-data-latent-360x640to4x4",
                       prefix="feats/unlabelled",
                       file_share_max=50)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    first_time = end - start
    
    start = time.time()
    batch = next(iter(loader)) 
    end = time.time()
    second_time = end - start
    
    x,w,y,z = batch
    print(f"Time to load first batch: {first_time:.2f}s")
    print(f"Time to load second batch: {second_time:.2f}s")
    print(f"Video shape: {x.shape}")
    print(f"Audio shape: {w.shape}")
    print(x.std())
    print(f"Mouse shape: {y.shape}") 
    print(f"Button shape: {z.shape}")

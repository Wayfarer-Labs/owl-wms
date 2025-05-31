from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F

import os
import random

class CoDLatentDataset(IterableDataset):
    def __init__(self, window_length = 120, root = "/home/shahbuland/cod_data/raw", add_optical_flow=True):
        super().__init__()

        self.window = window_length
        self.paths = []
        self.add_optical_flow = add_optical_flow

        for root_dir in os.listdir(root):
            splits_dir = os.path.join(root, root_dir, "splits")
            if not os.path.isdir(splits_dir):
                continue
                
            # Get all files in splits dir
            files = os.listdir(splits_dir)
            # Filter to just the base files (without _mouse or _buttons)
            base_files = [f for f in files if f.endswith("_rgblatent.pt")]
            
            for base_file in base_files:
                base_path = os.path.join(splits_dir, base_file)
                # Get just the numeric prefix from the filename
                base_name = base_file.split('_')[0]
                mouse_path = os.path.join(splits_dir, f"{base_name}_mouse.pt")
                buttons_path = os.path.join(splits_dir, f"{base_name}_buttons.pt") 
                of_path = os.path.join(splits_dir, f"{base_name}_flowlatent.pt")

                
                if os.path.exists(mouse_path) and os.path.exists(buttons_path) and (not self.add_optical_flow or os.path.exists(of_path)):
                    self.paths.append((base_path, mouse_path, buttons_path, of_path))
                else:
                    print(f"Missing files for {base_name}:")
                    if not os.path.exists(mouse_path):
                        print(f"  Missing mouse data: {mouse_path}")
                    if not os.path.exists(buttons_path):
                        print(f"  Missing button data: {buttons_path}")
                    if self.add_optical_flow and not os.path.exists(of_path):
                        print(f"  Missing optical flow data: {of_path}")
    
    def get_item(self):
        vid_path, mouse_path, btn_path, of_path = random.choice(self.paths)
        # Load tensors with memory mapping
        vid = torch.load(vid_path, map_location='cpu', mmap=True).float()
        mouse = torch.load(mouse_path, map_location='cpu', mmap=True) 
        buttons = torch.load(btn_path, map_location='cpu', mmap=True)

        if self.add_optical_flow:
            of = torch.load(of_path, map_location='cpu',mmap=True).float()

        # Get minimum length
        min_len = min(len(vid), len(mouse), len(buttons))

        # Get random starting point that allows for full window
        max_start = min_len - self.window
        window_start = random.randint(0, max_start)
        
        # Extract window slices
        vid_slice = vid[window_start:window_start+self.window]
        if self.add_optical_flow:
            of_slice = of[window_start:window_start+self.window]
            vid_slice = torch.cat([vid_slice, of_slice], dim = -1)
        
        mouse_slice = mouse[window_start:window_start+self.window]
        buttons_slice = buttons[window_start:window_start+self.window]

        return vid_slice, mouse_slice, buttons_slice # [n,c,h,w] [n,2], [n,n_buttons] respectively

    def __iter__(self):
        while True:
            yield self.get_item()

def collate_fn(x):
    # x is list of triples
    vids, mouses, buttons = zip(*x)
    vids = torch.stack(vids)      # [b,n,c,h,w]
    mouses = torch.stack(mouses)  # [b,n,2]
    buttons = torch.stack(buttons) # [b,n,n_buttons]
    return vids, mouses, buttons

def get_loader(batch_size, **data_kwargs):
    """
    Creates a DataLoader for the CoDDataset with the specified batch size
    
    Args:
        batch_size: Number of samples per batch
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = CoDLatentDataset(**data_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        prefetch_factor=1,
        num_workers=1
    )
    return loader

if __name__ == "__main__":
    import time
    loader = get_loader(64, root="../cod_data/BlackOpsColdWar", add_optical_flow=False)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    
    x,y,z = batch
    print(f"Time to load batch: {end-start:.2f}s")
    print(f"Video shape: {x.shape}")
    print(x.std())
    print(f"Mouse shape: {y.shape}") 
    print(f"Button shape: {z.shape}")
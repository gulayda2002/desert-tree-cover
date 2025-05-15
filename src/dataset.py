import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TileDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed tile .npz files.
    Each file contains 'image', 'mask', and 'meta'.
    """
    def __init__(self, tiles_dir, transform=None):
        self.tiles_dir = tiles_dir
        self.transform = transform
        # Gather all .npz files
        self.samples = []
        for scene_id in os.listdir(tiles_dir):
            scene_dir = os.path.join(tiles_dir, scene_id)
            if os.path.isdir(scene_dir):
                for fname in os.listdir(scene_dir):
                    if fname.endswith('.npz'):
                        self.samples.append(os.path.join(scene_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        image = data['image']  # shape: (4, H, W)
        mask = data['mask']    # shape: (H, W)
        meta = data['meta'].item()

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, meta

# Example usage:
# dataset = TileDataset('data/tiles')
# loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

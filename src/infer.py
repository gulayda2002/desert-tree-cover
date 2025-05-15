import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TileDataset
from model import UNet
from tqdm import tqdm

# Configuration
data_dir = "data/tiles"
output_dir = "data/results"
batch_size = 8
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint_path = "checkpoints/best_model.pth"

os.makedirs(output_dir, exist_ok=True)

# Load Dataset and DataLoader
dataset = TileDataset(data_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Load Model
model = UNet(in_channels=4, out_channels=2).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Inference Loop
with torch.no_grad():
    for images, masks, meta in tqdm(loader, desc="Inference"):  # meta is list of dicts
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # shape: (B, H, W)

        for i, p in enumerate(preds):
            tile_meta = meta[i]
            scene_id = tile_meta['scene_id']
            row = tile_meta['row_off']
            col = tile_meta['col_off']

            # Save prediction mask
            scene_out = os.path.join(output_dir, scene_id)
            os.makedirs(scene_out, exist_ok=True)
            out_path = os.path.join(scene_out, f"{scene_id}_{row}_{col}_pred.png")
            # Convert binary mask to PNG
            from PIL import Image
            img = Image.fromarray((p * 255).astype(np.uint8))
            img.save(out_path)

print("Inference completed.")

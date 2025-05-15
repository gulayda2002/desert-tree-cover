import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import TileDataset
from model import UNet

# Configuration
data_dir = "data/tiles"
checkpoint_dir = "checkpoints"
batch_size = 6
epochs = 20
learning_rate = 1e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset and DataLoader
dataset = TileDataset(data_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Model, Optimizer, Scheduler
model = UNet(in_channels=4, out_channels=2).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
def train():
    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch}_valloss{val_loss:.4f}.pth"))


if __name__ == "__main__":
    train()

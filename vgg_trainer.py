import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.csrnet_vgg import CSRNet
from utils.vgg_dataset import CrowdDataset
from datetime import datetime

# --- Konfigurasi ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
lr = 1e-5
num_epochs = 100
save_path = "csrnet_vgg.pt"
root = "dataset/part_A/train_data"

# --- Data ---
train_dataset = CrowdDataset(root)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
model = CSRNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training ---
best_loss = float('inf')
log_lines = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for img, density in train_loader:
        img, density = img.to(device), density.to(device)
        output = model(img)
        loss = criterion(output, density)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # --- Evaluasi Count Error (MAE) ---
    model.eval()
    with torch.no_grad():
        total_mae = 0.0
        for img, density in train_loader:
            img, density = img.to(device), density.to(device)
            output = model(img)
            pred_count = output.sum().item()
            gt_count = density.sum().item()
            total_mae += abs(pred_count - gt_count)

    mae = total_mae / len(train_loader)
    log = f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | MAE Count: {mae:.2f}"
    print(log)
    log_lines.append(log)

    # --- Simpan Model Terbaik ---
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), save_path)

# --- Simpan Log ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"train_log_{timestamp}.txt", "w") as f:
    f.write("\n".join(log_lines))

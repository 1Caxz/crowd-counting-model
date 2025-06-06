import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.csrnet_vgg import CSRNet  # teacher
from models.csrnet_mbv3 import CSRNetMobile  # student
from utils.dataset import CrowdDataset
from datetime import datetime

# --- Konfigurasi ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
lr = 1e-5
epochs = 100
root = "dataset/part_A/train_data"
save_path = "csrnet_mobile_kd.pt"

# --- Dataset ---
dataset = CrowdDataset(root)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
teacher = CSRNet().to(device)
teacher.load_state_dict(torch.load("csrnet_vgg.pt"))
teacher.eval()

# --- Update arsitektur CSRNetMobile agar output-nya 28x28 ---
class CSRNetMobileUpsampled(nn.Module):
    def __init__(self):
        super(CSRNetMobileUpsampled, self).__init__()
        from torchvision import models
        from torchvision.models import mobilenet_v3_small
        base_model = mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.frontend = base_model.features

        self.backend = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # upsample from 7x7 to 28x28

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.upsample(x)
        return x

student = CSRNetMobileUpsampled().to(device)
criterion = nn.MSELoss()
kd_criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(student.parameters(), lr=lr)

# --- Training ---
best_loss = float('inf')
log_lines = []

for epoch in range(epochs):
    student.train()
    epoch_loss = 0.0

    for img, gt_density in dataloader:
        img = img.to(device)
        gt_density = gt_density.to(device)

        with torch.no_grad():
            teacher_density = teacher(img)

        student_density = student(img)

        # Resize teacher output agar sesuai dengan student
        teacher_resized = F.interpolate(teacher_density, size=student_density.shape[2:], mode='bilinear', align_corners=False)

        mse_loss = criterion(student_density, gt_density)
        distill_loss = kd_criterion(
            torch.log_softmax(student_density, dim=1),
            torch.softmax(teacher_resized, dim=1)
        )
        loss = 0.7 * mse_loss + 0.3 * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # --- Evaluasi ---
    student.eval()
    total_mae = 0.0
    with torch.no_grad():
        for img, gt_density in dataloader:
            img, gt_density = img.to(device), gt_density.to(device)
            pred = student(img)
            pred_count = pred.sum().item()
            gt_count = gt_density.sum().item()
            total_mae += abs(pred_count - gt_count)

    mae = total_mae / len(dataloader)
    log = f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | MAE Count: {mae:.2f}"
    print(log)
    log_lines.append(log)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(student.state_dict(), save_path)

# --- Simpan Log ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"train_mobile_kd_log_{timestamp}.txt", "w") as f:
    f.write("\n".join(log_lines))

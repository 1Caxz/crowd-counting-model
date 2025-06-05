import torch
from torch import nn, optim
from models.csrnet_vgg import CSRNet
from utils.dataset import CrowdDataset
from torch.utils.data import DataLoader
from models.csrnet_mbv3 import CSRNetMobile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)

root = "dataset/part_A/train_data"
train_loader = DataLoader(CrowdDataset(root), batch_size=4, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for img, density in train_loader:
        img, density = img.to(device), density.to(device)
        out = model(img)
        loss = criterion(out, density)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "csrnet_vgg.pt")

model = CSRNetMobile().to(device)
torch.save(model.state_dict(), "csrnet_mobile.pt")

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
from PIL import Image
import numpy as np

class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'ground-truth')
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, 'GT_' + img_name.replace('.jpg', '.mat'))

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0][0][0][0][0]  # array of [x, y] points
        density_map = self.generate_density_map(np.array(img.shape[1:]), points)

        return img, torch.from_numpy(density_map).unsqueeze(0).float()

    def generate_density_map(self, shape, points):
        from scipy.ndimage import gaussian_filter
        density = np.zeros(shape, dtype=np.float32)
        for point in points:
            x = min(int(point[0]), shape[1] - 1)
            y = min(int(point[1]), shape[0] - 1)
            density[y, x] += 1
        density = gaussian_filter(density, sigma=2)
        density = cv2.resize(density, (28, 28))  # ← tambahkan ini
        return density

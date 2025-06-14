import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import cv2
import h5py


class CrowdDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")

        img_path = self.lines[index]

        img, target = load_data(img_path, self.train)

        # img = 255.0 * F.to_tensor(img)

        # img[0,:,:]=img[0,:,:]-92.8207477031
        # img[1,:,:]=img[1,:,:]-95.2757037428
        # img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform:
            img = self.transform(img)
        return img, target


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth').replace("IMG", "GT_IMG")
    img = Image.open(img_path).convert('RGB')
    # gt_file = h5py.File(gt_path)
    # target = np.asarray(gt_file['density'])
    with h5py.File(gt_path, 'r') as gt_file:
        target = np.asarray(gt_file['density']).astype(np.float32)
    if False:
        crop_size = (img.size[0] // 2, img.size[1] // 2)
        if random.randint(0, 9) <= -1:
            dx = random.randint(0, 1) * img.size[0] // 2
            dy = random.randint(0, 1) * img.size[1] // 2
        else:
            dx = int(random.random() * img.size[0] * 0.5)
            dy = int(random.random() * img.size[1] * 0.5)

        # crop_size = (img.size[0]/2, img.size[1]/2)
        # if random.randint(0, 9) <= -1:
        #     dx = int(random.randint(0, 1)*img.size[0]*1./2)
        #     dy = int(random.randint(0, 1)*img.size[1]*1./2)
        # else:
        #     dx = int(random.random()*img.size[0]*1./2)
        #     dy = int(random.random()*img.size[1]*1./2)

        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
    # target = cv2.resize(target, (target.shape[1]/8, target.shape[0]/8), interpolation=cv2.INTER_CUBIC)*64

    return img, target
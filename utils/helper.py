import numpy as np
from PIL import Image
import numpy as np
import h5py
import cv2
import torch
import shutil


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
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


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth'):
    torch.save(state, "build/"+task_id+"_"+filename)

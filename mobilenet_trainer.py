import os
import time
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.csrnet_mbv3 import MobileCSRNet
from models.csrnet_vgg import CSRNet
from utils.vgg_dataset import CrowdDataset
from utils.helper import *

TEACHER = "csrnet_vgg_B.pth"
STUDENTCKPT = "build/kd_B_mobilecsrnet_kd.pth"
TRAINJSON = "dataset/json/part_B_train.json"
TESTJSON = "dataset/json/part_B_test.json"
PART = "kd_B"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'lr': 1e-5,
        'batch_size': 1,
        'epochs': 300,
        'device': device,
        'print_freq': 20,
        'alpha': 0.5,
        'beta': 1.0,
        'gamma': 0.5,
        'delta': 0.1
    }

    teacher = CSRNet().to(device)
    checkpoint = torch.load(TEACHER, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['state_dict'])
    teacher.eval()

    student = MobileCSRNet().to(device)
    criterion_density = nn.MSELoss().to(device)
    criterion_count = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'])

    start_epoch = 0
    best_mae = float('inf')
    if STUDENTCKPT and os.path.isfile(STUDENTCKPT):
        print(f"=> loading student checkpoint '{STUDENTCKPT}'")
        checkpoint = torch.load(STUDENTCKPT, map_location=device)
        student.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_mae = checkpoint['best_prec1']

    with open(TRAINJSON) as f:
        train_list = json.load(f)
    with open(TESTJSON) as f:
        val_list = json.load(f)

    train_loader = DataLoader(CrowdDataset(train_list, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), train=True), batch_size=config['batch_size'], shuffle=True)

    val_loader = DataLoader(CrowdDataset(val_list, transform=transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), train=False), batch_size=1, shuffle=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(start_epoch, config['epochs']):
        train_loss = train_kd(train_loader, teacher, student, optimizer, criterion_density, criterion_count, config, epoch)
        mae = validate(val_loader, student, config, epoch)

        is_best = mae < best_mae
        best_mae = min(mae, best_mae)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'best_prec1': best_mae,
            'optimizer': optimizer.state_dict(),
        }, is_best, PART, 'mobilecsrnet_kd.pth')

        scheduler.step()


def train_kd(loader, teacher, student, optimizer, criterion_density, criterion_count, config, epoch):
    student.train()
    total_loss = 0.0

    for i, (img, target) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        img = img.to(config['device'])
        target = target.to(torch.float32).to(config['device']).unsqueeze(0)

        with torch.no_grad():
            teacher_output = teacher(img)
        student_output = student(img)

        loss_teacher = criterion_density(student_output, teacher_output)
        loss_count = criterion_count(student_output.sum(), target.sum())
        target_resized = nn.functional.interpolate(target, size=student_output.shape[2:], mode='bilinear', align_corners=False)
        loss_gt = criterion_density(student_output, target_resized)
        feat_t = teacher.get_features(img).detach()
        feat_s = student.get_features(img)
        feat_s_up = nn.functional.interpolate(feat_s, size=feat_t.shape[2:], mode='bilinear', align_corners=False)
        loss_feat = criterion_density(feat_s_up, feat_t)

        loss = config['alpha'] * loss_teacher + config['beta'] * loss_count + config['gamma'] * loss_gt + config['delta'] * loss_feat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"GT: {target.sum().item():.1f}, Pred: {student_output.sum().item():.1f}")

    return total_loss / len(loader)


def validate(loader, model, config, epoch):
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for img, target in tqdm(loader, desc="Validating"):
            img = img.to(config['device'])
            target = target.to(config['device']).unsqueeze(0)
            output = model(img)
            total_mae += abs(output.sum() - target.sum()).item()
    mae = total_mae / len(loader)
    print(f"Validation MAE: {mae:.3f}")

    with open("mae_log.txt", "a") as f:
        f.write(f"{epoch},{mae}\n")

    return mae


if __name__ == '__main__':
    main()

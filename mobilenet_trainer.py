import sys
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from models.csrnet_vgg import CSRNet
from models.csrnet_mbv3_final import MobileCSRNet
from utils.helper import save_checkpoint
from utils.mobile_dataset import CrowdDataset

# === Configuration Parameters (replaces argparse) ===


class Args:
    train_json = "dataset/json/part_A_train.json"
    test_json = "dataset/json/part_A_val.json"
    pre = "build/kd_A_mobilenet_final.pth"  # path to pretrained student model
    gpu = "0"
    task = "kd_A"
    original_lr = 1e-4
    lr = 1e-7
    batch_size = 1
    momentum = 0.95
    decay = 5e-4
    start_epoch = 0
    epochs = 400
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    workers = 4
    seed = int(time.time())
    print_freq = 30


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


args = Args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.manual_seed(args.seed)

# Load dataset lists
with open(args.train_json, 'r') as f:
    train_list = json.load(f)
with open(args.test_json, 'r') as f:
    val_list = json.load(f)

# Models
teacher = CSRNet().cuda()
teacher.load_state_dict(torch.load(
    "build/csrnet_vgg_A.pth", weights_only=False)['state_dict'])
teacher.eval()

student = MobileCSRNet().cuda()

# Loss and Optimizer
mse_loss = nn.MSELoss(reduction='mean').cuda()


def count_loss_fn(pred, gt):
    return torch.mean(torch.abs(pred.sum(dim=(1, 2, 3)) - gt.sum(dim=(1, 2, 3))))


def combined_loss(student_out, gt_map, teacher_out, mse_loss=nn.MSELoss(), weights=(0.6, 0.2, 0.2)):
    # Resize GT and teacher to match student
    target_size = student_out.shape[2:]

    gt_map = F.interpolate(gt_map, size=target_size,
                           mode='bilinear', align_corners=False)
    teacher_out = F.interpolate(
        teacher_out, size=target_size, mode='bilinear', align_corners=False)

    # === Normalize GT map after interpolation (preserve total count)
    gt_map_sum_before = gt_map.sum(dim=(2, 3), keepdim=True)
    gt_map = gt_map / (gt_map_sum_before + 1e-6) * gt_map_sum_before

    # === Optional: Match teacher density to GT total count (optional normalization)
    teacher_sum = teacher_out.sum(dim=(2, 3), keepdim=True)
    gt_sum = gt_map.sum(dim=(2, 3), keepdim=True)
    teacher_out = teacher_out * (gt_sum / (teacher_sum + 1e-6))

    # === Loss components
    # Density loss (GT)
    loss_gt = mse_loss(student_out, gt_map)
    # Distillation loss (teacher)
    loss_kd = mse_loss(student_out, teacher_out.detach())
    loss_count = ((student_out.sum(dim=(2, 3)) -
                  gt_map.sum(dim=(2, 3))) ** 2).mean()  # Count loss

    # === Combine
    total_loss = weights[0] * loss_gt + \
        weights[1] * loss_kd + weights[2] * loss_count
    return total_loss


# optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(student.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)
best_prec1 = 50000

# Resume student if provided
if args.pre and os.path.isfile(args.pre):
    print("=> loading student checkpoint '{}'".format(args.pre))
    checkpoint = torch.load(args.pre)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    student.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print("=> no checkpoint found at '{}'".format(args.pre))

train_loader = torch.utils.data.DataLoader(
    CrowdDataset(train_list, shuffle=True,
                 transform=transforms.Compose([
                     transforms.Resize((640, 360)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]),
                 train=True, seen=student.seen, batch_size=args.batch_size, num_workers=args.workers),
    batch_size=args.batch_size)

val_loader = torch.utils.data.DataLoader(
    CrowdDataset(val_list, shuffle=False,
                 transform=transforms.Compose([
                     transforms.Resize((640, 360)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]), train=False),
    batch_size=1)


def adjust_learning_rate(optimizer, epoch, steps, scales, original_lr):
    lr = original_lr
    for step, scale in zip(steps, scales):
        if epoch >= step:
            lr *= scale
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(epoch):
    student.train()
    losses = AverageMeter()
    print(f"Epoch {epoch}, training...")
    for i, (img, target) in enumerate(train_loader):
        img = img.cuda()
        # img = Variable(img)
        target = target.float().unsqueeze(0).cuda()
        # target = Variable(target)
        with torch.no_grad():
            teacher_out = teacher(img)
            # teacher_out = Variable(teacher_out)
        student_out = student(img)

        # loss = combined_loss(student_out, target, teacher_out, weights=(0.6, 0.2, 0.2))
        # teacher_out = F.interpolate(teacher_out, size=student_out.shape[2:], mode='bilinear', align_corners=False)
        # student_out = F.interpolate(
        #     student_out, size=teacher_out.shape[2:], mode='bilinear', align_corners=False)
        loss = mse_loss(student_out, teacher_out)
        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("Sum GT:", target.sum().item(), "Sum Teacher:", teacher_out.sum().item(), "Sum Student:", student_out.sum().item())
            print(
                f"Epoch {epoch} [{i}/{len(train_loader)}] - Loss: {losses.val:.4f} - Loss Avg: {losses.avg:.4f}")


def evaluate():
    student.eval()
    mae = 0.0
    for i, (img, target) in enumerate(val_loader):
        img = img.cuda()
        pred = student(img)
        pred_count = pred.sum().item()
        target = target.float().unsqueeze(0).cuda()
        gt_count = target.sum().item()
        print(f"GT count: {gt_count:.1f}, Predicted: {pred_count:.1f}")

        mae += abs(pred_count - gt_count)
    mae /= len(val_loader)
    print(f"[VAL] Epoch {epoch}: MAE = {mae:.3f}")
    return mae


for epoch in range(args.start_epoch, args.epochs):
    args.lr = adjust_learning_rate(
        optimizer, epoch, args.steps, args.scales, args.original_lr)
    train_one_epoch(epoch)
    mae = evaluate()
    scheduler.step(mae)
    is_best = mae < best_prec1
    best_prec1 = min(mae, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': student.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.task, "mobilenet_final.pth")

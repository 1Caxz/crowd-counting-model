import sys
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from models.csrnet_vgg import CSRNet
from models.csrnet_tinyvgg import TinyCSRNet
from models.csrnet_mbv3_final import MobileCSRNet
from utils.helper import save_checkpoint
from utils.mobile_dataset import CrowdDataset

# === Configuration Parameters (replaces argparse) ===
TEACHER = "build/csrnet_vgg_B.pth"
CHECKPOINT = "build/ckpt_kd_B_mobilenet_final.pth"
JSONTRAIN = "dataset/json/part_B_train.json"
JSONTEST = "dataset/json/part_B_val.json"
TASK = "kd_A"
GPU = "0"
ORIGINALLR = 1e-7
LR = 1e-7
BATCHSIZE = 1
MOMENTUM = 0.95
DECAY = 5e-4
STARTEPOCH = 0
EPOCHS = 400
STEPS = [-1, 1, 100, 150]
SCALES = [1, 1, 1, 1]
WORKERS = 4
SEED = int(time.time())
PRINTFREQ = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_prec1 = 50000


class AverageMeter(object):
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


def adjust_learning_rate(optimizer, epoch):
    LR = ORIGINALLR
    for i in range(len(STEPS)):
        scale = SCALES[i] if i < len(SCALES) else 1
        if epoch >= STEPS[i]:
            LR = LR * scale
            if epoch == STEPS[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR


def validate(val_list, student, teacher, criterion):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        CrowdDataset(val_list,
                     shuffle=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
                     ]),  train=False),
        batch_size=BATCHSIZE)

    student.eval()

    mae_student = 0
    mae_teacher = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        
        with torch.no_grad():
            student_out = student(img)
            # teacher_out = teacher(img)

        mae_student += abs(student_out.sum() -
                   target.sum().type(torch.FloatTensor).cuda())
        # mae_teacher += abs(teacher_out.sum() -
        #            target.sum().type(torch.FloatTensor).cuda())

    mae_student = mae_student/len(test_loader)
    # mae_teacher = mae_teacher/len(test_loader)
    print(f' * STUDENT MAE {mae_student:.3f}\t  TEACHER A MAE 132.894\t TEACHER B MAE 27.088')
    plt.imshow(student_out[0, 0].cpu().numpy(), cmap='jet')
    plt.axis('off')
    plt.title(f"GT: {target.sum().item():.1f}, Pred: {student_out.sum().item():.1f}")
    plt.savefig("inference/heatmaps/train.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    return mae_student


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        CrowdDataset(train_list,
                     shuffle=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
                     ]),
                     train=True,
                     seen=model.seen,
                     batch_size=BATCHSIZE,
                     num_workers=WORKERS),
        batch_size=BATCHSIZE)
    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), optimizer.param_groups[0]['lr']))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        student_out = model(img)
        
        with torch.no_grad():
            teacher_out = teacher(img)
            
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        student_out = F.interpolate(student_out, size=teacher_out.shape[2:], mode='bilinear', align_corners=False)
        loss = criterion(student_out, teacher_out.detach())
        # if epoch <= 70:
        #     loss = criterion(student_out, teacher_out.detach())
        # else:
        #     mse_loss = F.mse_loss(student_out, target)
        #     count_loss = torch.abs(student_out.sum() - target.sum())
        #     kd_loss = F.mse_loss(student_out, teacher_out.detach())
        #     loss = mse_loss + 0.1 * count_loss + 0.2 * kd_loss
        
        
        # count_diff = torch.abs(student_out.sum() - target.sum())
        # if target.sum() > 100:
        #     count_loss = 2.0 * count_diff
        #     kd_loss = 1.0 * F.mse_loss(student_out, teacher_out.detach())
        # else:
        #     count_loss = 0.5 * count_diff
        #     kd_loss = 0.3 * F.mse_loss(student_out, teacher_out.detach())
            
        
        # if target.sum() > 50:
        #     loss += 0.3 * count_loss

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINTFREQ == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data {data_time.val:.3f} ({data_time.avg:.3f})\t Loss {losses.val:.4f} ({losses.avg:.4f})\t GT {target.sum():.1f}\t Teacher {teacher_out.sum():.1f}\t Student {student_out.sum():.1f}\t")


with open(JSONTRAIN, 'r') as outfile:
    train_list = json.load(outfile)
with open(JSONTEST, 'r') as outfile:
    val_list = json.load(outfile)

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
torch.cuda.manual_seed(SEED)

teacher = CSRNet().cuda()
teacher.load_state_dict(torch.load(
    TEACHER, weights_only=False)['state_dict'])
teacher.eval()

model = TinyCSRNet()

model = model.cuda()

criterion = nn.MSELoss(reduction='sum')

# optimizer = torch.optim.SGD(model.parameters(), LR,
#                             momentum=MOMENTUM,
#                             weight_decay=DECAY)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

if CHECKPOINT:
    if os.path.isfile(CHECKPOINT):
        print("=> loading checkpoint '{}'".format(CHECKPOINT))
        checkpoint = torch.load(CHECKPOINT)
        STARTEPOCH = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(CHECKPOINT, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(CHECKPOINT))

for epoch in range(STARTEPOCH, EPOCHS):

    # adjust_learning_rate(optimizer, epoch)

    train(train_list, model, criterion, optimizer, epoch)
    prec1 = validate(val_list, model, teacher, criterion)
    scheduler.step(prec1)

    is_best = prec1 < best_prec1
    best_prec1 = min(prec1, best_prec1)
    print(' * best Student MAE {mae:.3f} '
          .format(mae=best_prec1))
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': CHECKPOINT,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, TASK, "mobilenet_final.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

class MobileCSRNet(nn.Module):
    def __init__(self):
        super(MobileCSRNet, self).__init__()
        self.seen = 0
        # Backbone from pretrained MobileNetV3
        base_model = mobilenet_v3_large(weights='DEFAULT')
        self.frontend = base_model.features  # Output: [B, 960, H/32, W/32]
        
        # Reduce channel 960 → 512
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(size=(80, 45), mode='bilinear', align_corners=False)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)         # [B, 960, H/32, W/32]
        x = self.reduce_conv(x)      # [B, 512, H/32, W/32]
        x = self.upsample(x)         # [B, 512, H/8, W/8]
        x = self.backend(x)          # [B, 64, H/8, W/8]
        x = self.output_layer(x)     # [B, 1, H/8, W/8]
        return x

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
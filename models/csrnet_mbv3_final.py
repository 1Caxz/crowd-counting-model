# mobile_csrnet.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


def make_backend_layers(cfg, in_channels=960, dilation=True):
    layers = []
    d_rate = 2 if dilation else 1
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                           padding=d_rate, dilation=d_rate)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


class MobileCSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(MobileCSRNet, self).__init__()
        self.seen = 0

        # Frontend: MobileNetV3 Large
        mobilenet = mobilenet_v3_large(
            weights='DEFAULT' if load_weights else None)
        self.frontend = mobilenet.features  # Produces 960 channels

        # Backend: CSRNet-style
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_backend_layers(self.backend_feat, in_channels=960)
        
        self.adapter = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Output: 1-channel density map
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.frontend(x)          # Output: [B, 960, H/32, W/32]
        x = self.adapter(x)
        x = self.backend(x)           # Output: [B, 64, H/32, W/32]
        x = self.output_layer(x)      # Output: [B, 1, H/32, W/32]
        return x

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileCSRNet(nn.Module):
    def __init__(self):
        super(MobileCSRNet, self).__init__()

        mobilenet = mobilenet_v3_large(weights='DEFAULT')
        # Output of mobilenet.features is [B, 960, H/32, W/32] for large model
        self.frontend = mobilenet.features

        self.backend = nn.Sequential(
            # Change the input channels from 576 to 960 to match the frontend output
            nn.Conv2d(960, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def get_features(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = self.upsample(x)
        return x

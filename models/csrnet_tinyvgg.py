import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class TinyCSRNet(nn.Module):
    def __init__(self):
        super(TinyCSRNet, self).__init__()
        self.seen = 0
        mobilenet = mobilenet_v3_small(weights='DEFAULT')

        # Potong frontend agar output-nya resolusi H/8
        self.frontend = nn.Sequential(*list(mobilenet.features.children())[:10])  # Output: [B, 96, H/8, W/8]

        # Backend dengan tambahan Conv + BatchNorm
        self.backend = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # atau nn.Softplus()
        )

        # Upsample untuk menyamakan resolusi dengan ground truth
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.frontend(x)       # [B, 96, H/8, W/8]
        x = self.backend(x)        # [B, 32, H/8, W/8]
        x = self.output_layer(x)   # [B, 1, H/8, W/8]
        x = self.upsample(x)       # [B, 1, H/4, W/4] — sesuaikan dengan ground truth
        return x

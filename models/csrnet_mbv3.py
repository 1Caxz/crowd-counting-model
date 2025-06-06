import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import models

class CSRNetMobile(nn.Module):
    def __init__(self):
        super(CSRNetMobile, self).__init__()
        base_model = mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.frontend = base_model.features
        
        self.backend = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

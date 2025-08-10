import torch
import torch.nn as nn

# U‑Net building blocks for patch‐based segmentation

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = double_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, bilinear=True):
        super().__init__()
        self.in_conv  = double_conv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // factor)

        self.up1      = Up(1024, 512 // factor, bilinear)
        self.up2      = Up(512, 256 // factor, bilinear)
        self.up3      = Up(256, 128 // factor, bilinear)
        self.up4      = Up(128, 64, bilinear)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.out_conv(x)


def build_unet_model(in_channels: int = 3,
                     num_classes: int = 4,
                     device: torch.device = torch.device('cpu')) -> UNet:
    """
    Build a UNet model for patch segmentation and move it to device.

    :param in_channels: number of input channels (default=3)
    :param num_classes: number of segmentation classes (default=4)
    :param device: torch device
    :return: UNet instance
    """
    model = UNet(in_channels, num_classes)
    return model.to(device)

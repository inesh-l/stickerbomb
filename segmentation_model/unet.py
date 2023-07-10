import torch.nn as nn
import unet_components as components

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = components.InputConvolution(n_channels, 64)
        self.down1 = components.DownConvolution(64, 128)
        self.down2 = components.DownConvolution(128, 256)
        self.down3 = components.DownConvolution(256, 512)
        self.down4 = components.DownConvolution(512, 512)
        self.up1 = components.UpConvolution(1024, 256)
        self.up2 = components.UpConvolution(512, 128)
        self.up3 = components.UpConvolution(256, 64)
        self.up4 = components.UpConvolution(128, 64)
        self.outc = components.OutputConvolution(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)



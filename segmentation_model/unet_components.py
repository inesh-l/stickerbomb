import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class InputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputConvolution, self).__init__()
        self.conv = DoubleConvolution(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)
    
class DownConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConvolution, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.mpconv(x)

class UpConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvolution, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConvolution(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputConvolution, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.conv(x)
    
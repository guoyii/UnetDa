import torch.nn.functional as F
from model_function import *
from torch import nn

class UnetDa(nn.Module):
    def __init__(self, args):
        super(UnetDa, self).__init__()
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.bilinear = args.bilinear
        self.k_size = args.k_size

        self.inc = DoubleConv(self.input_channels, 64, k_size=self.k_size)
        self.down1 = Down(64, 128, k_size=self.k_size)
        self.down2 = Down(128, 256, k_size=self.k_size)
        self.down3 = Down(256, 512, k_size=self.k_size)
        self.down4 = Down(512, 512, k_size=self.k_size)
        self.up1 = Up(1024, 256, self.bilinear, k_size=self.k_size)
        self.up2 = Up(512, 128, self.bilinear, k_size=self.k_size)
        self.up3 = Up(256, 64, self.bilinear, k_size=self.k_size)
        self.up4 = Up(128, 64, self.bilinear, k_size=self.k_size)
        self.outc = OutConv(64, self.output_channels)

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
        logits = self.outc(x)
        return logits



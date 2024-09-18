from CNN.Block import *
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down4 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        self.conv1 = block(24)
        self.conv2 = block(48)
        self.conv3 = block(96)
        self.conv4 = block(192)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up4 = nn.Sequential(nn.Conv2d(24, 3, kernel_size=3, stride=1, padding=1))

        self.mix1 = CGAFusion(192, reduction=8)
        self.mix2 = CGAFusion(96, reduction=8)
        self.mix3 = CGAFusion(48, reduction=4)

    def forward(self,x):
        x_pre = x

        x_down1 = self.down1(x)
        skip1 = x_down1
        x_down1 = self.conv1(x_down1)

        x_down2 = self.down2(x_down1)
        skip2 = x_down2
        x_down2 = self.conv2(x_down2)

        x_down3 = self.down3(x_down2)
        skip3 = x_down3
        x_down3 = self.conv3(x_down3)

        x_down4 = self.down4(x_down3)

        x1 = self.conv4(x_down4)

        x_level3_mix = self.mix1(x_down4, x1)
        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.conv3(x_up1)


        x_level2_mix = self.mix2(x_down3, x_up1)
        # skip4 = x_level2_mix
        x_up1 = self.up2(x_level2_mix)
        x_up1 = self.conv2(x_up1)

        x_level2_mix = self.mix3(x_down2, x_up1)
        skip4 = x_level2_mix
        x_up2 = self.up3(x_level2_mix)
        x_up2 = self.conv1(x_up2)

        out = self.up4(x_up2) + x_pre

        return out,skip1,skip2,skip4

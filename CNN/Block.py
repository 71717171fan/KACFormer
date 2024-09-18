import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class block(nn.Module):
    def __init__(self,in_channel):
        super(block,self).__init__()

        self.op1 = nn.Conv2d(in_channel,in_channel,1,1,0)
        self.op2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.op3 = nn.Conv2d(in_channel, in_channel, 5, 1, 2)
        self.op4 = nn.Conv2d(in_channel, in_channel, 7, 1, 3)

        self.act = nn.ReLU(inplace=True)
        self.calayer = CALayer(in_channel)
        self.palayer = PALayer(in_channel)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel*4,in_channel*2,3,1,1),
            nn.Conv2d(in_channel*2,in_channel,3,1,1),
        )
    def forward(self,x):

        op1 = self.op1(x)
        op1 = self.act(op1) + x
        op1 = self.op1(op1)
        op1 = self.calayer(op1)
        op1 = self.palayer(op1) + x

        op2 = self.op2(x)
        op2 = self.act(op2) + x
        op2 = self.op2(op2)
        op2 = self.calayer(op2)
        op2 = self.palayer(op2) + x

        op3 = self.op3(x)
        op3 = self.act(op3) + x
        op3 = self.op3(op3)
        op3 = self.calayer(op3)
        op3 = self.palayer(op3) + x

        op4 = self.op4(x)
        op4 = self.act(op4) + x
        op4 = self.op4(op4)
        op4 = self.calayer(op4)
        op4 = self.palayer(op4) + x

        y = torch.cat([op1,op2,op3,op4],1)
        y = self.conv(y) + x

        return y



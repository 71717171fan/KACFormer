from Base_Network import *
from seg.seg_model import *
from seg.seg_utils import *
from seg.deeplab import *
from seg.seg_model import deeplabv3_resnet50
from seg.seg_model import deeplabv3plus_resnet50
from seg.seg_model import deeplabv3_resnet101
from seg.seg_model import deeplabv3plus_resnet101
from seg.seg_model import deeplabv3_mobilenet
from seg.seg_model import deeplabv3plus_mobilenet
import torchvision.models as models
# from meta import *
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
class Skip(nn.Module):
    def __init__(self, input_nc, output_nc, RDB_num=19, seg_dim=32):
        super(Skip, self).__init__()
        self.seg_conv = nn.Sequential(
                RDB(RDB_num),
                nn.Conv2d(RDB_num, seg_dim, 1, bias=True)
        )
        self.convR = nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0)
        self.conv_seg = nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(seg_dim,3,1,1,1)
        self.corse_conv = nn.Sequential(
            Depthwise_separable_conv(3, seg_dim, kernel_size=3,padding=1, bias=False)
        )

        self.conv = nn.Conv2d(seg_dim, output_nc, kernel_size=1, stride=1, padding=0)

        self.combine_convs = nn.Sequential(
            nn.Conv2d(seg_dim,seg_dim,3,1,1,bias=True),
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, output_nc, 3, 1, 1, bias=True),
        )

    def forward(self, x, y):

        model_name = 'deeplabv3plus_mobilenet'  # Adjust this based on your segmentation model

        model_map = {
            'deeplabv3_resnet50': deeplabv3_resnet50,
            'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
            'deeplabv3_resnet101': deeplabv3_resnet101,
            'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet
        }

        num_classes = 19  # Adjust based on your segmentation task
        model = model_map[model_name](num_classes=num_classes, output_stride=16)

        checkpoint = torch.load('D:/ws/Transformer_CNN/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = nn.DataParallel(model)
        seg = model(self.conv_seg(y))
        # print(seg.shape)
        l = self.seg_conv(seg)
        # print(l.shape)
        r = self.corse_conv(self.convR(x))
        # print(r.shape)
        f = r + l
        # print(f.shape)
        y = self.combine_convs(f)
        # print(y.shape)
        return y

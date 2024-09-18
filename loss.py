import torch.nn as nn
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import models
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '3': 'relu1_2',  # 第一个卷积块的第二个ReLU层
            '8': 'relu2_2',  # 第二个卷积块的第二个ReLU层
            '15': 'relu3_3',  # 第三个卷积块的第三个ReLU层
            '22': 'relu4_3'  # 第四个卷积块的第三个ReLU层
        }

    def forward(self, x, layer='22'):
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                if name == layer:
                    return x
        return x

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor().to(device)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # 提取目标图像和生成图像的特征
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        # 计算特征的L1损失
        loss = self.criterion(features_x, features_y)
        return loss
from models.Base_Network import *
from models.seg.seg_model import *
from models.seg.seg_utils import *
from models.seg.deeplab import *
from models.seg.seg_model import deeplabv3_resnet50
from models.seg.seg_model import deeplabv3plus_resnet50
from models.seg.seg_model import deeplabv3_resnet101
from models.seg.seg_model import deeplabv3plus_resnet101
from models.seg.seg_model import deeplabv3_mobilenet
from models.seg.seg_model import deeplabv3plus_mobilenet
import torchvision.models as models
# from meta import *
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image

class Skip(nn.Module):
    def __init__(self, input_nc, output_nc, RDB_num=19, seg_dim=32):
        super(Skip, self).__init__()
        # if input_nc==48:
        #   self.seg_conv = nn.Sequential(
        #       RDB(RDB_num),
        #       nn.Conv2d(RDB_num, seg_dim, 3, 2,1,bias=True)
        #   )
        # else:
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
            # nn.Tanh()
        )

    # def forward(self, x, y, weighted_input1, weighted_input2):
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
        # seg = model(y)
        seg = model(self.conv_seg(y))
        l = self.seg_conv(seg)
        r = self.corse_conv(self.convR(x))

        # weighted_input1 = weighted_input1.to(device)
        # weighted_input2 = weighted_input2.to(device)

        # f = weighted_input1 * r + weighted_input2 * l
        f = r + l
        y = self.combine_convs(f)

        return y

import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, features):
        super(FeatureExtractor, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
            # 'avgpool': "avgpool"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        out_3 = None
        out_8 = None
        out_15 = None
        out_22 = None
        # out_avgpool = None

        for name, module in self.features._modules.items():
            x = module(x)
            if name == '3':
                out_3 = x
            elif name == '8':
                out_8 = x
            elif name == '15':
                out_15 = x
            elif name == '22':
                out_22 = x
            # elif name == 'avgpool':
            #     out_avgpool = x

        target_size = (224, 224)
        out_8 = F.interpolate(out_8, size=target_size, mode='bilinear', align_corners=False)
        out_15 = F.interpolate(out_15, size=target_size, mode='bilinear', align_corners=False)
        out_22 = F.interpolate(out_22, size=target_size, mode='bilinear', align_corners=False)


        return out_3, out_8, out_15, out_22


def preprocess(image):
    # print(image.shape)
    image_tensor = image.squeeze(0)
    # print(image_tensor.shape)
    # 现在 pil_images 是一个包含所有转换后 PIL 图像的列表
    pil_image = to_pil_image(image_tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_image).unsqueeze(0)


# 提取特征函数
def extract_features(image, model):
    # 加载并预处理图像
    # 将图像输入模型并提取特征
    features = model(image)

    return features

class OALayer(nn.Module):
    ##################################其中num_ops代表所需要加参数的位置数目,channel代表输入通道数，根据实际改变！！！！！！！##################################
    def __init__(self, num_ops=4, channel=960, k=1):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

# 拼接特征
# def concat_features(fea1, fea2, fea3, fea4, fea5):
#     # 在第一维度上拼接特征
#     concatenated_features = torch.cat((fea1, fea2, fea3, fea4, fea5), dim=1)
#     return concatenated_features
class pre(nn.Module):

    def __init__(self,a,b):
        super(pre, self).__init__()
        self.a = a
        self.b = b

    def forward(self, img):


        y = np.float(self.a) * img + self.b
        y[y > 255] = 255
        y = np.round(y)
        img_bright = y.astype(np.uint8)

        img_norm = img / 255.0  # 注意255.0得采用浮点数
        img_gamma = np.power(img_norm, 2.5) * 255.0
        img_gamma = img_gamma.astype(np.uint8)

        r, g, b = cv2.split(img)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]
        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        balance_img = cv2.merge([b, g, r])

        balance_img = transforms.ToTensor()(balance_img)
        img_bright = transforms.ToTensor()(img_bright)
        img_gamma = transforms.ToTensor()(img_gamma)
        img = transforms.ToTensor()(img)

        hazef = torch.cat([img,balance_img, img_bright, img_gamma], 0)

        height = hazef.shape[1]
        width = hazef.shape[2]
        hazef = hazef.expand(1, 12, height, width)


        return  hazef

# meta = torch.load("D:/ws/Transformer_CNN/T_result/dehaze_80.pth", map_location=lambda storage, loc: storage)
prepro = pre(0.5,0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class weight_generate(nn.Module):
    def __init__(self):
        super(weight_generate,self).__init__()

        vgg16 = models.vgg16(pretrained=True)

        self.weights = []
        self.attention = OALayer()

        self.feature_extractor = FeatureExtractor(vgg16.features)
    def forward(self,x):
      # 提取特征,x是输入的张量
        image = preprocess(x)
      # fea1, fea2, fea3, fea4 = self.feature_extractor(image)

        # lr_patch = Variable(x, requires_grad=False).cuda(0)
        #
        # unloader = transforms.ToPILImage()
        # image = lr_patch.cpu().clone()
        # image = image.squeeze(0)
        # haze = unloader(image)
        # x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)
        #
        # pre_haze = prepro(x).to(device)
        # start_time = time.time()

        # fea1,fea2,fea3,fea4,fea5,fea6 = meta(pre_haze)
    # 输出提取的特征
        fea1, fea2, fea3, fea4 = self.feature_extractor(image)
        haze_features = torch.cat((fea1, fea2, fea3, fea4), 1)

        # print(f'After SwinTransformerUNet x1 shape: {haze_features.shape}')

        weights = F.softmax(self.attention(haze_features), dim=-1)

        return weights
# class weight_generate1(nn.Module):
#     def __init__(self):
#         super(weight_generate1,self).__init__()
#
#         vgg16 = models.vgg16(pretrained=True)
#
#         self.weights = []
#         self.attention = OALayer1()
#
#         self.feature_extractor = FeatureExtractor(vgg16.features)
#     def forward(self,x):
#     # 提取特征,x是输入的张量
#         image = preprocess(x)
#         fea1, fea2, fea3, fea4 = self.feature_extractor(image)
#
#     # 输出提取的特征
#
#         haze_features = torch.cat((fea1, fea2, fea3, fea4), 1)
#
#         # print(f'After SwinTransformerUNet x1 shape: {haze_features.shape}')
#
#         weights = F.softmax(self.attention(haze_features), dim=-1)
#
#         # return weights
#         return weights
# class OALayer1(nn.Module):
#     ##################################其中num_ops代表所需要加参数的位置数目,channel代表输入通道数，根据实际改变！！！！！！！##################################
#     def __init__(self, num_ops=14, channel=960, k=1):
#         super(OALayer1, self).__init__()
#         self.k = k
#         self.num_ops = num_ops
#         self.output = k * num_ops
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ca_fc = nn.Sequential(
#             nn.Linear(channel, self.output * 2),
#             nn.ReLU(),
#             nn.Linear(self.output * 2, self.k * self.num_ops))
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = y.view(x.size(0), -1)
#         y = self.ca_fc(y)
#         y = y.view(-1, self.k, self.num_ops)
#         return y
# class fea(nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg16 = models.vgg16(pretrained = True)
#         self.feature = FeatureExtractor(vgg16.features)
#
#     def forward(self,x):
#         image = preprocess(x)
#         fea1, fea2, fea3, fea4 = self.feature(image)
#         return fea1,fea2,fea3,fea4
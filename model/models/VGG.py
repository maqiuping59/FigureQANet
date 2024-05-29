# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 10:59
# @Author  : MaQiuping
# @FileName: VGG.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import torchvision.models
from torchvision.models import vgg16
import torch.nn as nn
import torch


class VGGModule(nn.Module):
    def __init__(self,device):
        super(VGGModule,self).__init__()
        vggModel = vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        vggModel.to(device)
        vggModel.eval()
        self.features = nn.Sequential(*list(vggModel.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x




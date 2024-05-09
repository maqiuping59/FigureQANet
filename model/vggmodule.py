# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 10:59
# @Author  : MaQiuping
# @FileName: vggmodule.py
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
        model.to(device)
        model.eval()
        self.features = nn.Sequential(*list(vggModel.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


model = VGGModule()
x = torch.randn(1, 3, 224, 224)
outputs = model(x)
print(outputs.shape)


# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : Ma Qiu ping
# @FileName: model.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from vggmodule import VGGModule


class ChartQuestionModel(nn.Module):
    def __init__(self,num_answers):
        super(ChartQuestionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = VGGModule(device=self.device).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('../transformers_bert/microsoft/codebert')
        self.text_encoder = AutoModel.from_pretrained('../transformers_bert/microsoft/codebert')
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.text_adapter = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU()
        )

        # 定义融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128+64, 256),
            nn.ReLU(),
            nn.Linear(256, num_answers)
        )

    def forward(self, chart, question):
        # 图表特征提取
        # chart_features = self.chart_encoder(chart)
        chart_features = self.vgg(chart)
        print(chart_features.shape)
        chart_features = chart_features.view(chart_features.size(0), -1)

        # 问题特征提取
        tokens = self.tokenizer(question, return_tensors='pt')
        question_features = self.text_encoder(tokens).last_hidden_state  # N*768

        # 特征融合
        combined_features = torch.cat((chart_features, question_features), dim=1)
        output = self.fusion_layer(combined_features)

        return output




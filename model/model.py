# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: model.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import os
import torch
import torch.nn as nn


class ChartQuestionModel(nn.Module):
    def __init__(self):
        super(ChartQuestionModel, self).__init__()
        # 定义图表特征提取器
        self.chart_encoder=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 更多层...
        )

        # 定义问题特征提取器
        self.question_encoder=nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=300),
            nn.GRU(input_size=300, hidden_size=128, num_layers=2, batch_first=True),
            # 更多层...
        )

        # 定义融合层
        self.fusion_layer=nn.Sequential(
            nn.Linear(128+64, 256),
            nn.ReLU(),
            nn.Linear(256, num_answers)
        )

    def forward(self, chart, question):
        # 图表特征提取
        chart_features=self.chart_encoder(chart)
        chart_features=chart_features.view(chart_features.size(0), -1)

        # 问题特征提取
        question_features, _=self.question_encoder(question)
        question_features=question_features[:, -1, :]

        # 特征融合
        combined_features=torch.cat((chart_features, question_features), dim=1)
        output=self.fusion_layer(combined_features)

        return output
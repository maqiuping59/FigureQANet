# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : Ma Qiu ping
# @FileName: FigureQANet.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from einops.layers.torch import Rearrange
from einops import rearrange
from model.models.VGG import VGGModule


class ChartQuestionModel(nn.Module):
    def __init__(self, num_answers, embed_dim, answer_vocab_num, pretrained_model):
        super(ChartQuestionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chart_encoder = VGGModule(device=self.device).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.text_adapter = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        self.chart_adapter = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(512, embed_dim),
        )

        # Modal-Fuse
        self.attentionFuse = nn.MultiheadAttention(embed_dim, num_heads=8)


    def forward(self, chart, question):
        with torch.no_grad():
            tokens = self.tokenizer(question, return_tensors='pt', padding="max_length",
                                    max_length=30, add_special_tokens=True)
            attention_mask = tokens['attention_mask']
            question_features = self.text_encoder(tokens.input_ids,
                                                  attention_mask=attention_mask).last_hidden_state  # N*768
            chart_features = self.chart_encoder(chart)
        # chart_features = chart_features.view(chart_features.size(0), -1)
        chart_features = self.chart_adapter(chart_features)
        print("chart_features.shape", chart_features.shape)
        print("question_features.shape", question_features.shape)
        exit()
        # fuse
        combined_features = torch.cat((chart_features, question_features), dim=1)
        output = self.fusion_layer(combined_features)

        return output

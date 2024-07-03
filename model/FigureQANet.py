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
    def __init__(self,embed_dim, answer_vocab_num, pretrained_model,dropout=0.2):
        super(ChartQuestionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.fuseLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5880, 10000,bias=True),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(10000, answer_vocab_num, bias=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.ReLU(inplace=True),
        )



    def forward(self, chart, question):
        with torch.no_grad():
            tokens = self.tokenizer(question, return_tensors='pt', padding="max_length",
                                    max_length=30, add_special_tokens=True)
            attention_mask = tokens['attention_mask']
            tokens  =tokens.to(self.device)
            attention_mask = attention_mask.to(self.device)
            question_features = self.text_encoder(tokens.input_ids,
                                                  attention_mask=attention_mask).last_hidden_state  # N*768
            chart_features = self.chart_encoder(chart)
        # chart_features = chart_features.view(chart_features.size(0), -1)
        chart_features = self.chart_adapter(chart_features)
        question_features = self.text_adapter(question_features)

        # query=chart key=value=question
        query = chart_features.permute(1, 0, 2)
        value = key = question_features.permute(1, 0, 2)
        # value = question_features.permute(1, 0, 2)
        attn_chart_que, attn_weights_chart_que = self.attentionFuse(query, key, value)
        attn_chart_que = attn_chart_que.permute(1, 0, 2)
        # print(attn_chart_que.size(),attn_weights_chart_que.size()) torch.Size([8, 196, 256]) torch.Size([8, 196, 30])

        # query=question key=value=value
        query = question_features.permute(1, 0, 2)
        value = key = chart_features.permute(1, 0, 2)
        attn_que_chart, attn_weights_que_chart = self.attentionFuse(query, key, value)
        attn_que_chart = attn_que_chart.permute(1, 0, 2)
        # print(attn_que_chart.size(),attn_weights_que_chart.size()) torch.Size([8, 30, 256]) torch.Size([8, 30, 196])

        attn = torch.einsum("bij,bkj->bik", attn_chart_que, attn_que_chart)
        attn = attn.div(attn.norm(p=2, dim=1, keepdim=True))
        attn_weights = attn_weights_que_chart.permute(0, 2, 1) + attn_weights_chart_que

        outputs = self.fuseLayer(attn)
        outputs = torch.softmax(outputs, dim=1)

        return outputs,attn_weights

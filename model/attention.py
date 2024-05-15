# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 7:28
# @Author  : MaQiuping
# @FileName: attention.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59


import torch
import torch.nn as nn
import torch.nn.functional as F


# RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)
        return output, hidden


model = RNNModel(input_size=10, hidden_size=50, output_size=1000)
input_tensor = torch.randn(1, 1, 10)

hidden = torch.zeros(1, 1, 50)

output, hidden = model(input_tensor, hidden)




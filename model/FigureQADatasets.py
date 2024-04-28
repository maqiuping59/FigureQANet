# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:25
# @Author  : MaQiuping
# @FileName: FigureQADatasets.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import os
import json
from torch.utils.data import Dataset,DataLoader

class ChartQuestionDataset(Dataset):
    def __init__(self, charts, questions, answers):
        self.charts = charts
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        chart = self.charts[index]
        question = self.questions[index]
        answer = self.answers[index]
        return chart, question, answer



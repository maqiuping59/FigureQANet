# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59


from torch.utils.data import DataLoader

from model.FigureQADatasets import ChartQuestionDataset
from model.model import ChartQuestionModel

# 假设你已经有了图表、问题和答案的数据
charts = ...  # 图表数据
questions = ...  # 问题数据
answers = ...  # 答案数据

dataset = ChartQuestionDataset(charts, questions, answers)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

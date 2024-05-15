# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59

import argparse
import yaml
from torch.utils.data import DataLoader
from data.FigureQADatasets import ChartQADataset, DVQADataset
from model.model import ChartQuestionModel
from torch.optim import Adam
import torch.nn as nn


def train(model,dataloader,args):
    criterion = nn.CrossEntropyLoss()
    model = ChartQuestionModel(num_answers=1)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    best_acc = 0

    for epoch in range(args.num_epochs):
        model.train()
        if args.train.Parallel:
            nn.DataParallel(model).to(args.device)
            for i,batch in enumerate(dataloader):
                questions = batch["question"]
                answers = batch["answers"]
                images = batch["image"]
                outputs = model(questions, images)
                outputs = outputs.logits()
                loss = criterion()

                if i % args.train.print_freq == 0:
                    print("===")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    train_dataset_ChartQA=ChartQADataset("./data/ChartQA", "train", )
    train_dataloader_ChartQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)

    val_dataset_ChartQA=ChartQADataset("./data/ChartQA", "val", )
    val_dataloader_ChartQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)

    train_dataset_DVQA=DVQADataset("./data/ChartQA", "train", )
    train_dataloader_DVQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)

    val_dataset_DVQA=DVQADataset("./data/ChartQA", "val", )
    val_dataloader_DVQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)

    model = ChartQuestionModel(784,4)

    train(model,args)


if __name__ == '__main__':
    main()



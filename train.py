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
from model.FigureQANet import ChartQuestionModel
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm


def train(dataloader,args):
    criterion = nn.CrossEntropyLoss()
    model = ChartQuestionModel(num_answers=1)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    best_acc = 0

    model.train()

    start_epoch = 0
    if args.resume.resume_train and args.resume.resume_epoch > 0:
        start_epoch = args.resume.resume_epoch
        model = torch.load(args.saved_model)
    for epoch in range(start_epoch, args.num_epochs+1):
        if args.train.Parallel:
            nn.DataParallel(model).to(args.device)
            scheduler.step()
            for i, batch in tqdm(enumerate(dataloader)):
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

    train_dataset_ChartQA=ChartQADataset("./data/ChartQA", "train")
    train_dataloader_ChartQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)

    val_dataset_ChartQA=ChartQADataset("./data/ChartQA", "val")
    val_dataloader_ChartQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)

    train_dataset_DVQA=DVQADataset("./data/ChartQA", "train", phase="train")
    train_dataloader_DVQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)

    val_dataset_DVQA=DVQADataset("./data/ChartQA", "val",phase="train")
    val_dataloader_DVQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)

    train(train_dataset_ChartQA,args)


if __name__ == '__main__':
    main()



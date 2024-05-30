# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59

import argparse
import yaml
from data.FigureQADatasets import get_dvqa_loader
from data.getAnswerSet import DVQA_answer_vocab
from model.FigureQANet import ChartQuestionModel
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import evaluate
import pprint
from easydict import EasyDict

import time


def train(args):
    answer_vocab_num = len(DVQA_answer_vocab)
    dvqa_config = args.train.datasets.DVQA
    image_dir = dvqa_config.imagePath
    qa_train = dvqa_config.train.qaPath
    qa_val = dvqa_config.val.qaPath
    dataloader = get_dvqa_loader(image_dir=image_dir, qa_train=qa_train, qa_val=qa_val,
                                 num_workers=args.train.num_workers)
    criterion = nn.CrossEntropyLoss()
    model = ChartQuestionModel(num_answers=2, answer_vocab_num=answer_vocab_num, pretrained_model=args.pretrain)
    optimizer = Adam(model.parameters(), lr=args.train.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.step_size, gamma=args.train.gamma)
    best_acc = 0

    start_epoch = 0
    if args.resume.resume_train and args.resume.resume_epoch > 0:
        start_epoch = args.resume.resume_epoch
        model = torch.load(args.saved_model)
    for epoch in range(start_epoch, args.train.num_epochs+1):
        if args.train.Parallel:
            nn.DataParallel(model).to(args.device)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    scheduler.step()
                else:
                    model.eval()
                scheduler.step()
                for i, batch in tqdm(enumerate(dataloader[phase])):
                    questions = batch["question"]
                    answers = batch["answer"]
                    images = batch["image"]
                    labels = batch["answer_id"]
                    outputs = model(images, questions)
                    outputs = outputs.logits()
                    # loss = criterion()

                    if i % args.train.print_freq == 0:
                        print("===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = EasyDict(config)
    train(config)


if __name__ == '__main__':
    main()



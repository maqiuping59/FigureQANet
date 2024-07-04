# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59

import argparse
import os

import yaml
from data.FigureQADatasets import get_dvqa_loader
from data.FigureQADatasets import DVQADataset, BaseTransform
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
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import random
from termcolor import colored


torch.manual_seed(42)
random.seed(42)



def train(args):
    logging.basicConfig(filename="train.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    writer = SummaryWriter(args.logs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answer_vocab_num = len(DVQA_answer_vocab)
    dvqa_config = args.train.datasets.DVQA
    image_dir = dvqa_config.imagePath
    qa_train = dvqa_config.train.qaPath
    qa_val = dvqa_config.val.qaPath
    # load data

    resize = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = BaseTransform(resize, mean, std)
    train_dataset = DVQADataset(image_dir,qapath=qa_train,phase="train",transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train.batch_size,shuffle=True)

    val_dataset = DVQADataset(image_dir, qapath=qa_val, phase="val", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train.batch_size, shuffle=False)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    criterion = nn.CrossEntropyLoss()
    model = ChartQuestionModel(embed_dim=args.model.embed_dim,answer_vocab_num=answer_vocab_num,
                               pretrained_model=args.pretrain, dropout=args.train.dropout)
    optimizer = Adam(model.parameters(), lr=args.train.learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.step_size, gamma=args.train.gamma)
    best_acc = 0

    accuracy = evaluate.load("./accuracy")
    start_epoch = 0

    # resume training
    if args.resume.resume_train:
        ckpt = os.path.join(args.train.saveDir,'checkpoints', 'last.pt')
        ckpt = torch.load(ckpt,map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['weights'], strict=False) # strict 允许跳过不匹配的参数
        optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1
        logging.info("resume train from epoch {}".format(colored("{}".format(start_epoch), "green", attrs=['bold'])))

    logging.info("start training:")
    for epoch in range(start_epoch, args.train.num_epochs):
        for phase in ['train', 'val']:
            logging.info("<=========>epoch:{}->{}<=========>".format(colored(epoch, "yellow", attrs=['bold']),
                                                                     colored(phase, "green", attrs=['bold'])))
            print("<=========>epoch:{}->{}<=========>".format(colored(epoch, "yellow", attrs=['bold']),
                                                                     colored(phase, "green", attrs=['bold'])))
            epoch_accuracy = evaluate.load("./accuracy")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            batch_step_size = len(dataloader[phase].dataset) / args.train.batch_size
            for batch_idx, batch in tqdm(enumerate(iter(dataloader[phase]))):
                optimizer.zero_grad()
                questions = batch["question"]
                images = batch["image"]
                labels = batch["answer_id"].cuda()
                if torch.cuda.is_available():
                    images = images.to(device)
                    model.to(device)
                result,attn_weights = model(images,questions)
                outputs = result.logit()
                _,indices = torch.max(outputs,1)
                loss = criterion(result,labels)
                metrix = accuracy.compute(predictions=indices, references=labels)
                epoch_accuracy.add_batch(predictions=indices, references=labels)
                acc = metrix['accuracy']
                msg = "{}| EPOCH:{}/{}|STEP:{}/{}|Loss:{:.4f},Accuracy:{:.2f}".format(phase.upper(),
                                                                                      epoch+1,args.train.num_epochs,
                                                                                      batch_idx,int(batch_step_size),
                                                                                      loss.item(),acc)
                print(msg)
                logging.info(msg)
                loss.backward()
                optimizer.step()

                if acc>best_acc:
                    best_acc = acc
                    if not os.path.exists(args.train.saveDir):
                        os.makedirs(args.train.saveDir)
                    torch.save(model.state_dict(), os.path.join(args.train.saveDir,"best.ptn"))

            accu = epoch_accuracy.compute()
            writer.add_scalar('Accuracy/{}'.format(phase), accu["accuracy"], epoch)
            print("{}| EPOCH:{}|Accuracy:{:.2f}".format(phase.upper(),epoch+1,accu["accuracy"]))
            logging.info("{}| EPOCH:{}|Accuracy:{:.2f}".format(phase.upper(),epoch+1,accu["accuracy"]))
        # scheduler.step()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = EasyDict(config)
    pprint.pprint(config)
    train(config)


if __name__ == '__main__':
    main()



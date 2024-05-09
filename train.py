# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59

import argparse
import yaml
# from torch.utils.data import DataLoader
# from data.FigureQADatasets import ChartQADataset, BaseTransformer, DVQADataset
from data.FigureQADatasets import BaseTransformer
# from model.model import ChartQuestionModel


def train(model,args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    baseTransformer=BaseTransformer(resize=224, mean=mean, std=std)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    print(config)

    # train_dataset_ChartQA=ChartQADataset("./data/ChartQA", "train", )
    # train_dataloader_ChartQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)
    #
    # val_dataset_ChartQA=ChartQADataset("./data/ChartQA", "val", )
    # val_dataloader_ChartQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)
    #
    # train_dataset_DVQA=DVQADataset("./data/ChartQA", "train", )
    # train_dataloader_DVQA=DataLoader(train_dataset_ChartQA, batch_size=32, shuffle=True)
    #
    # val_dataset_DVQA=DVQADataset("./data/ChartQA", "val", )
    # val_dataloader_DVQA=DataLoader(val_dataset_ChartQA, batch_size=32, shuffle=False)
    #
    # model = ChartQuestionModel(784,4)
    #
    # train(model,args)


if __name__ == '__main__':
    main()



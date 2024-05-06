# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:25
# @Author  : MaQiuping
# @FileName: FigureQADatasets.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import os
import json
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class ChartQADataset(Dataset):
    def __init__(self, datapath, phase):
        super(ChartQADataset,self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.image_path = os.path.join(self.datapath, self.phase, "png")

        augmented = json.load(open(os.path.join(self.datapath, self.phase,self.phase+"_augmented.json"), "r"))
        human = json.load(open(os.path.join(self.datapath, self.phase,self.phase+"_human.json"), "r"))
        self.data = augmented + human

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = self.data[index]["query"]
        answer = self.data[index]["label"]
        image_name = self.data[index]["imgname"]
        image_path = os.path.join(str(self.image_path),image_name)
        # if os.path.exists(image_path):
        #     return query,answer,image_path
        return query,answer,image_path


class DVQADataset(Dataset):
    def __init__(self, imagepath, qapath):
        super(DVQADataset,self).__init__()
        self.image_path = imagepath
        self.qa_path = qapath

        self.qalist = json.load(open(self.qa_path,"r"))
        print(len(self.qalist))
        print(self.qalist[0])

    def __len__(self):
        return len(self.qalist)

    def __getitem__(self,index):
        query = self.qalist[index]["question"]
        template_id = self.qalist[index]["template_id"]
        answer = self.qalist[index]["answer"]
        image_name = self.qalist[index]["image"]


dataset = DVQADataset(imagepath="./DVQA/images", qapath="./DVQA/qa/train_qa.json")





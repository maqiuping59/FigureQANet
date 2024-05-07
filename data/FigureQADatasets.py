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
import transformers
from torchvision import transforms


class BaseTransformer():
    def __init__(self,resize,mean,std):
        self.basetransform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    def __call__(self,img):
        return self.basetransform(img)


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
        self.image_list = os.listdir(self.image_path)

    def __len__(self):
        return len(self.qalist)

    def __getitem__(self,index):
        query = self.qalist[index]["question"]
        template_id = self.qalist[index]["template_id"]
        answer = self.qalist[index]["answer"]
        image_name = self.qalist[index]["image"]
        answer_box = self.qalist[index]["answer_bbox"]
        image_path = os.path.join(str(self.image_path),image_name)

        return query,template_id,answer,image_path,answer_box


class FigureQADataset(Dataset):
    def __init__(self, image_path,qa_path):
        super(FigureQADataset,self).__init__()
        self.image_list = os.listdir(image_path)
        self.qa_list = json.load(open(qa_path,"r"))['qa_pairs']

    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self,index):
        query = self.qa_list[index]
        return query


dataset = FigureQADataset(image_path="./FigureQA/figureqa-train1-v1/png",
                          qa_path="./FigureQA/figureqa-train1-v1/qa_pairs.json")






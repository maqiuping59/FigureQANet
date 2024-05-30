# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:25
# @Author  : MaQiuping
# @FileName: FigureQADatasets.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import transformers
from torchvision import transforms
import torch
from data.getAnswerSet import DVQA_answer_vocab


class BaseTransform:
    def __init__(self, resize, mean, std):
        self.baseTransform = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.baseTransform[phase](img)


class ChartQADataset(Dataset):
    def __init__(self, datapath, phase,transform=None):
        super(ChartQADataset, self).__init__()
        self.datapath=datapath
        self.phase=phase
        self.image_path=os.path.join(self.datapath, self.phase, "png")
        self.transform = transform

        augmented = json.load(open(os.path.join(self.datapath, self.phase, self.phase+"_augmented.json"), "r",encoding="utf-8"))
        human = json.load(open(os.path.join(self.datapath, self.phase, self.phase+"_human.json"), "r"))
        self.data = augmented+human

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query=self.data[index]["query"]
        answer=self.data[index]["label"]
        image_name=self.data[index]["imgname"]
        image_path=os.path.join(str(self.image_path), image_name)
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image, self.phase)
        # if os.path.exists(image_path):
        #     return query,answer,image_path
        return {"question": query, "answer": answer, "image": image,"answer_id":ChartQA_answer_vocab[answer]}


class DVQADataset(Dataset):
    def __init__(self, imagepath, qapath,phase,transform=None):
        super(DVQADataset, self).__init__()
        self.image_path=imagepath
        self.qa_path=qapath
        self.phase = phase
        self.transform = transform

        self.qalist=json.load(open(self.qa_path, "r",encoding="utf-8"))
        self.image_list=os.listdir(self.image_path)

    def __len__(self):
        return len(self.qalist)

    def __getitem__(self, index):
        query=(self.qalist[index]["question"]).lower()
        template_id=self.qalist[index]["template_id"]
        answer=(self.qalist[index]["answer"]).lower()
        image_name=self.qalist[index]["image"]
        answer_box=self.qalist[index]["answer_bbox"]
        image_path=os.path.join(str(self.image_path), image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image,self.phase)
        answer_id = DVQA_answer_vocab[answer]
        return {"question":query, "template_id":template_id, "answer":answer, "image":image,"answer_id":answer_id}


class FigureQADataset(Dataset):
    def __init__(self, image_path, qa_path,phase,transform=None):
        super(FigureQADataset, self).__init__()
        self.image_path=image_path
        self.image_list=os.listdir(self.image_path)
        self.qa_list=json.load(open(qa_path, "r"))['qa_pairs']
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self, index):
        query = self.qa_list[index]
        image_name = str(query['image_index'])
        image_path = os.path.join(str(self.image_path), image_name+'.png')
        question = query['question_string']
        answer = query['answer']
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image,self.phase)
        return {"question": question, "image": image, "answer": answer}


# resize = (224,224)
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# transform = BaseTransform(resize=resize,mean=mean,std=std)
# dataset = DVQADataset(r"E:\FigureQANet\data\DVQA\images", r"E:\FigureQANet\data\DVQA\qa\preprocessedtrain_qa.json",phase="train",transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)


def get_dvqa_loader(image_dir,qa_train,qa_val,batch_size=8,num_workers=0):
    resize = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize=resize, mean=mean, std=std)
    vqa_dataset = {
        'train': DVQADataset(
            imagepath=image_dir,
            qapath=qa_train,
            phase="train",
            transform=transform),
        'valid': DVQADataset(
            imagepath=image_dir,
            qapath=qa_val,
            phase="val",
            transform=transform)}

    data_loader = {
        phase: DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader



# -*- coding: utf-8 -*-
# @Time    : 2024/4/28 18:22
# @Author  : MaQiuping
# @FileName: val.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59


from data.FigureQADatasets import DVQADataset,BaseTransform
from torch.utils.data import Dataset, DataLoader
resize = (224, 224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)

dataset = DVQADataset(imagepath="./data/DVQA/images",qapath="data/DVQA/qa/preprocessedtrain_qa.json",phase="train",transform=transform)

dataloader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0)
batch_step_size = len(dataloader.dataset)/128

for idx,batch in enumerate(dataloader):
    print(idx,int(batch_step_size))




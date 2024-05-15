# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 21:26
# @Author  : MaQiuping
# @FileName: FigureCaptionDatasets.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import json

from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pylab as plt
import torch


class ChartQACaptionDatasets(Dataset):
    def __init__(self, data_path,phase):
        super(ChartQACaptionDatasets,self).__init__()
        self.data_path = data_path
        self.phase = phase
        self.image_path = os.path.join(self.data_path,self.phase,'png')
        self.annotation_path = os.path.join(self.data_path,self.phase,'annotations')
        self.table_path = os.path.join(self.data_path,self.phase,'tables')

        self.image_list = os.listdir(str(self.image_path))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        image_name = self.image_list[index]
        image_path = os.path.join(str(self.image_path),image_name)
        annotation_path = os.path.join(str(self.annotation_path),image_name.replace("png","json"))
        table_path = os.path.join(str(self.table_path),image_name.replace("png","csv"))
        return image_path,annotation_path,table_path


# class DVQACaptionDataset(Dataset):
#     def __init__(self,image_path,metadata_path):
#         super(DVQACaptionDataset,self).__init__()
#         self.image_path = image_path
#         self.metadata_path = metadata_path
#         self.image_list = os.listdir(str(self.image_path))
#         self.metadata_list = json.load(open(self.metadata_path,"r"))
#         print(metadata_path[0])
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self,index):
#         image_name = self.image_list[index]


class SimChartCaption(Dataset):
    def __init__(self,image_path,table_path):
        super(SimChartCaption,self).__init__()
        self.image_path = image_path
        self.table_path = table_path

        self.imageList = os.listdir(self.image_path)

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self,index):
        image_name = self.imageList[index]
        image_path = os.path.join(str(self.image_path),image_name)
        table_path = os.path.join(str(self.table_path),image_name.replace("png","csv"))
        table = pd.read_csv(table_path)

        return image_path,table










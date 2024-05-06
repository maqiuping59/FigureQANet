# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 21:26
# @Author  : MaQiuping
# @FileName: FigureCaptionDatasets.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59

from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd


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


class DVQACaptionDataset(Dataset):
    def __init__(self,image_path,metadata_path):
        super(DVQACaptionDataset,self).__init__()
        self.image_path = image_path
        self.metadata_path = metadata_path

        self.image_list = os.listdir(str(self.image_path))





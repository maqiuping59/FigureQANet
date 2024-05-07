# -*- coding: utf-8 -*-
# @Time    : 2024/5/7 8:45
# @Author  : MaQiuping
# @FileName: preprocess.py
# @Software: PyCharm
# @Blog    ：https://github.com/maqiuping59
import json
import os
from tqdm import tqdm

imageList = os.listdir("./images")

for file in os.listdir("qa/"):
    new_qas = []
    qas = json.load(open(os.path.join("./qa",file),'r'))
    for qa in tqdm(qas):
        if qa['image'] in imageList:
            new_qas.append(qa)
    with open("./qa/preprocessed"+file,"w") as newfile:
        json.dump(new_qas,newfile)

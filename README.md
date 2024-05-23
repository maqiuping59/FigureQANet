### FigureQANet

图标问答模型

项目结构

```bash
├─data
│  ├─ChartQA
│  │  ├─test
│  │  │  ├─annotations
│  │  │  ├─png
│  │  │  └─tables
│  │  ├─train
│  │  │  ├─annotations
│  │  │  ├─png
│  │  │  └─tables
│  │  └─val
│  │      ├─annotations
│  │      ├─png
│  │      └─tables
│  ├─DVQA
│  │  ├─qa
│  │  ├─DVQA_dataset
│  │  ├─images
│  │  └─metadata
│  ├─SimChart9K
│  │  ├─table
│  │  └─png
│  ├─FigureQA
│  │  ├─figureqa-train1-v1
│  │  │  └─png
│  │  ├─figureqa-validation1-v1.tar
│  │  │  └─png
│  │  ├─figureqa-validation2-v1
│  │  │  └─validation2
│  │  │      └─png
│  │  └─test
│  │      └─no_annot_test1
│  │          └─png
│  └─__pycache__
├─transformers_bert
│  ├─microsoft
│  │  └─codebert
│  └─pix2struct
├─model
│  ├─utils
│  └─__pycache__
├─utils
├─config
├─logs
└─run
```

#### 数据集

- ChartQA
- DvQA
- FigureQA
- SimChart9K

#### 训练

```bash
python train.py --config './config/train_config.yaml'
```


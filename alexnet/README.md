## AlexNet 代码复现

### 数据准备

- [AlexNet 预训练模型](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth) ：保存至 `./data`
- [猫狗数据集](https://www.kaggle.com/c/dogs-vscats-redux-kernels-edition/data) ：保存至 `./data`

### 代码结构

```
.
├── README.md
├── data
│   ├── alexnet-owt-4df8aa71.pth
│   ├── golden_retriever.jpg
│   ├── imagenet1000.json
│   ├── imagenet_classnames.txt
│   ├── test
│   ├── tiger_cat.jpg
│   └── train
├── results                // 可视化文件存储目录
├── src                    // 核心代码目录
│   ├── inference.py       // ImageNet 1000 类 AlexNet 测试
│   ├── visualization.py   // AlexNet 卷积核可视化，特征图可视化
│   └── train.py           // 猫狗数据集上训练 AlexNet
└── tools                  // 工具类目录
    ├── __init__.py
    └── dataset.py         // 自定义数据集加载类
```
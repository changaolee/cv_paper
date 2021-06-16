import os
import random
from torch.utils.data import Dataset
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        :param data_dir: 数据集路径
        :param mode: 两种模式 train 和 valid
        :param split_n: 训练集划分比例
        :param rng_seed: 随机种子
        :param transform: 数据处理方法
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)  # 在这里做 transform，转为 tensor 等

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):
        """
        存储所有图片路径和标签
        在 DataLoader 中通过 index 读取样本
        :return:
        """
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        # 设置随机种子，固定图片排序，保证训练集与验证集不变
        random.seed(self.rng_seed)
        random.shuffle(img_names)

        img_labels = [0 if name.startswith('cat') else 1 for name in img_names]

        # 计算训练集切分点
        split_idx = int(len(img_labels) * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        path_img_set = [os.path.join(self.data_dir, name) for name in img_set]
        data_info = [(name, label) for name, label in zip(path_img_set, label_set)]

        return data_info

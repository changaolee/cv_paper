import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from alexnet.tools.dataset import CatDogDataset
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :param vis_model:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def ten_crop_transform(crops):
    normalize = transforms.Normalize(norm_mean, norm_std)
    to_tensor = transforms.ToTensor()
    return torch.stack([normalize(to_tensor(crop)) for crop in crops])


if __name__ == '__main__':
    # config
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    checkpoint_dir = os.path.join(BASE_DIR, "..", "checkpoint")
    data_dir = os.path.join(BASE_DIR, "..", "data", "train")

    num_classes = 2
    start_epoch = 0

    CHECKPOINT_INTERVAL = 5  # 可自行修改
    MAX_EPOCH = 3  # 可自行修改
    BATCH_SIZE = 128  # 可自行修改
    LR = 0.001  # 可自行修改
    LR_DECAY_STEP = 1  # 可自行修改

    # ============================== step 1/5 数据 ================================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),  # 与 (256, 256) 区别：先短边缩放至 256，长边再等比例缩放
        transforms.CenterCrop(256),  # 中心裁剪 256x256
        transforms.RandomCrop(224),  # 随机裁剪 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: ten_crop_transform(crops)),
    ])

    # 构建 Dataset 实例
    train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建 DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================== step 2/5 模型 ================================
    alexnet_model = get_model(path_state_dict, False)

    # 修改最后一层的输出为 2 个，实现二分类任务
    num_in_features = alexnet_model.classifier._modules["6"].in_features
    alexnet_model.classifier._modules["6"] = nn.Linear(num_in_features, num_classes)

    alexnet_model.to(device)

    # ============================ step 3/5 损失函数 ==============================
    criterion = nn.CrossEntropyLoss()

    # ============================= step 4/5 优化器 ===============================
    flag = False  # 可自行修改，是否冻结卷积层
    if flag:
        fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是 FC 层参数的内存地址
        base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},
            {'params': alexnet_model.classifier.parameters(), 'lr': LR}
        ], lr=LR, momentum=0.9)
    else:
        optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

    # 设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.1)

    # ============================== step 5/5 训练 ================================
    train_curve = list()
    valid_curve = list()

    for epoch in range(start_epoch, MAX_EPOCH):

        # train the model
        correct, total = 0., 0.
        alexnet_model.train()

        for i, data in enumerate(train_loader):
            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().numpy()

            # 打印训练信息
            train_curve.append(loss.item())
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i + 1, len(train_loader), loss.item(), correct / total))

        scheduler.step()  # 更新学习率

        # 保存模型
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                'model_state_dict': alexnet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            path_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_{}_epoch.pkl'.format(epoch))
            torch.save(checkpoint, path_checkpoint)

        # validate the model
        correct_val, total_val, loss_val = 0., 0., 0.
        alexnet_model.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 由于验证集数据使用了 TenCrop，需要 view 转换成 b,c,h,w 才能输入模型
                bs, n_crops, c, h, w = inputs.size()  # [4, 10, 3, 224, 224]
                outputs = alexnet_model(inputs.view(-1, c, h, w))
                outputs_avg = outputs.view(bs, n_crops, -1).mean(1)

                loss = criterion(outputs_avg, labels)

                _, predicted = torch.max(outputs_avg.data, 1)
                total_val += labels.size(0)
                correct_val += torch.eq(predicted, labels).sum().numpy()

                loss_val += loss.item()

            loss_val_mean = loss_val / len(valid_loader)
            valid_curve.append(loss_val_mean)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iter = len(train_loader)
    # 由于 valid 中记录的是 epoch_loss，需要对记录点进行转换到 iterations
    valid_x = np.arange(1, len(valid_curve) + 1) * train_iter
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

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


if __name__ == '__main__':
    # config
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger_cat.jpg")
    log_dir = os.path.join(BASE_DIR, "..", "results")

    # -------------------------- 卷积核可视化 ----------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")

    # load model
    alexnet_model = get_model(path_state_dict, False)

    vis_max_num = 1  # 配置可视化层的数量
    for i, sub_module in enumerate(alexnet_model.modules()):
        # 过滤非卷积层
        if not isinstance(sub_module, nn.Conv2d):
            continue
        if i >= vis_max_num:
            break

        kernels = sub_module.weight
        c_out, c_in, k_h, k_w = tuple(kernels.shape)

        # 拆分 channel
        for o_idx in range(c_out):
            # 获得 (3, h, w), 但是 make_grid 需要 b,c,h,w，这里拓展 c 维度变为(3, 1, h, w)
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_in)
            writer.add_image('{}_conv_layer_split_in_channel'.format(i), kernel_grid, global_step=o_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
        writer.add_image('{}_all'.format(i), kernel_grid, global_step=620)
        writer.close()

        print("{}_conv_layer shape:{}".format(i, tuple(kernels.shape)))

    # -------------------------- 特征图可视化 ----------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature_map")

    # 数据
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    # 读取图片并进行 transform
    img_rgb = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_rgb)
    img_tensor.unsqueeze_(0)  # c,h,w -> b,c,h,w

    # forward
    conv_layer_1 = alexnet_model.features[0]
    feature_map_1 = conv_layer_1(img_tensor)

    # 预处理
    feature_map_1.transpose_(0, 1)  # b,c,h,w = (1, 64, 55, 55) -> (64, 1, 55, 55)
    feature_map_1_grid = vutils.make_grid(feature_map_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', feature_map_1_grid, global_step=620)
    writer.close()
import json
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载分类名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def process_img(path):
    """
    图片数据读取
    :param path:
    :return:
    """
    # hard code 基于 ImageNet 统计得来的均值和标准差
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 将图片数据转换为模型读取的形式
    img_rgb = Image.open(path).convert('RGB')
    img_tensor = inference_transform(img_rgb)

    # c,h,w -> b,c,h,w
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


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
    # path_img = os.path.join(BASE_DIR, "..", "data", "golden_retriever.jpg")
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger_cat.jpg")
    path_classnames = os.path.join(BASE_DIR, "..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "imagenet_classnames.txt")

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    alexnet_model = get_model(path_state_dict, True)

    # 3/5 inference: tensor -> vector
    with torch.no_grad():
        time_tic = time.time()
        outputs = alexnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15 + idx * 30, "top {}:{}".format(idx + 1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()

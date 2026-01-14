import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.utils as vutils
import torch as th

# 在通道维度(维度1)上应用softmax
softmax_helper = lambda x: F.softmax(x, 1)

# 应用sigmoid函数
sigmoid_helper = lambda x: torch.sigmoid(x)

# 初始化权重
class InitWeights_He(object):
    def __init__(self, neg_slope = 1e-2):
        # 负斜率，用于LeakyReLU等激活函数
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            # 初始化nn.Conv3d，nn.Conv2d，nn.ConvTranspose2d，nn.ConvTranspose3d四种类型的网络
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

# 转torch.Tensor
def maybe_to_torch(d):
    if isinstance(d, list):
        # 列表，对每一个列表中的非torch.Tensor继续调用maybe_to_torch
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        # 转为torch.Tensor
        d = torch.from_numpy(d).float()
    return d

# 转移到GPU
def to_cuda(data, non_blocking = True, gpu_id = 0):
    if isinstance(data, list):
        # 列表，列表中的每一个元素移动到gpu上，non_blocking异步传输
        data = [i.cuda(gpu_id, non_blocking = non_blocking) for i in data]
    else:
        # 移动到gpu上
        data = data.cuda(gpu_id, non_blocking = non_blocking)
    return data

# 空的操作符
class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # 实现STAPLE（simultaneous truth and performance level estimation）算法
    # 多个分割结果中提取共识结果
    # 该函数接收形状为(n, c, h, w)的张量，表示n个分割结果
    # n表示不同分割结果的数量，c表示类别数，h和w表示图像的高度和宽度
    
    # 共识的分割结果，初始为所有分割结果的平均
    mvres = mv(a)

    # 计算初始的差异
    gap = 0.4

    # 迭代直到差异小于0.02
    if gap > 0.02:
        for i, s in enumerate(a):

            # 对每个分割结果s，计算s * mvres（逐元素相乘）
            r = s * mvres

            # 将所有结果拼接成新的张量
            res = r if i == 0 else torch.cat((res, r), 0)

        # 计算新的共识结果nres为res的平均值
        nres = mv(res)

        # 计算新旧共识结果之间的平均绝对误差gap
        gap = torch.mean(torch.abs(mvres - nres))

        # 更新共识结果mvres = nres和输入a = res
        mvres = nres
        a = res
    return mvres

# 功能：合并视盘(disc)和视杯(cup)分割结果
# 将两个灰度图合并为一个
# 反转颜色（255 - res）：黑色变白色
# 返回PIL图像对象
# 视盘和视杯都是医学上的概念，合并结果用于进一步分析
def allone(disc, cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup, 0, 1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

# 功能：计算dice系数（分割评估指标）
# Dice = 2 × (交集) / (并集)
# 用于衡量分割结果与真实标签的相似度
def dice_score(pred, targs):
    # 二值化预测结果
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()

# 功能：沿批次维度计算平均值，保持维度不变
# 输入形状：(b, c, h, w)
# 输出形状：(1, c, h, w)
def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim = True) / b

# 功能：将PyTorch张量转换为numpy图像数组
# 从GPU移到CPU并转换为numpy
# 维度转换：从(C, H, W)或(B, C, H, W)转换为(H, W, C)或(B, H, W, C)
def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

# 功能：保存张量为图像文件
# 如果是3通道：直接保存
# 如果不是3通道：取最后一个通道，复制为3通道保存（灰度图）
# 使用torchvision.utils.save_image
def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

# 功能：标准化张量（均值为0，标准差为1）
# 计算均值和标准差
# 执行标准化：(x - mean) / std
def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s

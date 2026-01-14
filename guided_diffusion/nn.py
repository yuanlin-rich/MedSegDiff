"""
Various utilities for neural networks.
"""
# 神经网络模块

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # silu激活函数
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    # 转为float32的group norm，再转回原数据类型
    # group norm，将channels分为num_groups组，对每组进行归一化，group norm只在自己的样本内进行
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    # 卷积
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def layer_norm(shape, *args, **kwargs):
    # layer norm，相当于group norm的组数等于1的情况
    return nn.LayerNorm(shape, *args, **kwargs)

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    # 线性层
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    # 局部空间取平均池，减少空间尺寸
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate = 0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    # ema训练，让目标参数更接近源参数
    # 就是在target_params和source_params之间做指数加权平均
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha = 1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 零初始化
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    # 缩放
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    # 保留第0维（batch维），对其他所有维求均值
    # 计算张量除批次维度外的所有维度的平均值
    # 得到的结果形状是[batch_size]，即只保留批次维度
    return tensor.mean(dim = list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # group norm
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # 时间嵌入
    # 输入：timesteps形状为(N,)，表示N个时间步； dim表示嵌入维度； max_period控制频率范围
    # 输出：形状为(N, dim)的时间嵌入张量
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start = 0, end = half, dtype = th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        # 处理奇数维度，如果是奇数维度，补一个0
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# 梯度检查点技术，用于内存优化
# 问题背景
# 神经网络前向传播会保存所有中间激活值用于反向传播
# 内存占用随网络深度线性增长，限制模型规模
# 解决方案
# 正常训练（默认情况）：
# 前向传播：保存中间激活值（activation），不是梯度
# 反向传播：利用保存的激活值计算梯度
# 开启梯度检查点：
# 前向传播：不保存中间激活值（或只保存部分关键点）
# 反向传播：需要梯度时重新计算相关部分的前向传播（得到激活值），然后计算梯度

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function             # 函数
        ctx.input_tensors = list(args[:length])     # 函数输入
        ctx.input_params = list(args[length:])      # 其他参数
        with th.no_grad():
            # 不记录计算图的前向传播，不追踪梯度，不构建计算图
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # detach()：从原计算图分离，避免影响之前的梯度计算
        # requires_grad_(True)：需要梯度，因为要重新计算
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            # 重新计算前向传播
            # 创建浅拷贝，避免原地修改问题
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        
        # 计算梯度
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused = True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

"""
Helpers for distributed training.
"""
# 分布式训练相关，但是被硬编码为本地gpu相关

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    # 设置分布式训练环境
    if dist.is_initialized():
        # 已经初始化则直接返回
        return
    
    if not args.multi_gpu:
        # 单GPU模式下，设置可见GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev

    # 选择后端：GPU用nccl，CPU用gloo
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        # 获取本地机器的主机名
        hostname = socket.gethostbyname(socket.getfqdn())

    # 关键：硬编码为单进程设置
    # 本地回环地址
    os.environ["MASTER_ADDR"] = '127.0.1.1'     #comm.bcast(hostname, root=0)

    # 进程排名固定为0
    os.environ["RANK"] = '0'                    #str(comm.rank)

    # 总进程数固定为1
    os.environ["WORLD_SIZE"] = '1'              #str(comm.size)

    # 动态分配端口号（避免端口冲突）
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定到任意可用端口
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)

    # 初始化进程组（实际上只是单进程）
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    # 返回当前可用的设备，优先使用GPU
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # 只有rank 0（第0个进程）会读取文件
    mpigetrank = 0
    if mpigetrank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None    
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

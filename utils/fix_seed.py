import torch
import numpy as np
import random


def same_seeds(seed):
    """设置随机种子"""
    # 设置 Python 内置随机模块的种子
    random.seed(seed)
    # 设置 Numpy 的种子
    np.random.seed(seed)
    # 设置 Torch 的种子
    torch.manual_seed(seed)
    # 如果 cuda 可用，则设置相应的随机种子
    if torch.cuda.is_available():
        print("\ncuda可用, 并设置相应的随机种子 Seed = ", seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 禁用 cuDNN 的自动调速模式，即不使用最快的卷积算法
    torch.backends.cudnn.benchmark = False
    # 启用 cuDNN 的确定性模式，即只使用确定性的卷积算法
    torch.backends.cudnn.deterministic = True


def set_random_seed(seed, use_cuda=False):
    """
    Set the random seed for various libraries.

    Parameters:
    - seed: Integer random seed value to be used.
    - use_cuda: Boolean flag to determine if cuda random seed should be set.
    """
    try:
        # Setting Python's built-in random seed
        random.seed(seed)
        # Setting numpy's random seed
        np.random.seed(seed)
        # Setting PyTorch's random seed
        torch.manual_seed(seed)

        if use_cuda and torch.cuda.is_available():
            print("CUDA is available, setting CUDA device random seeds.seed=", seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Disable the cuDNN automatic algorithm selection
        torch.backends.cudnn.benchmark = False
        # Enable the cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = True
    except RuntimeError as e:
        print(f"An error occurred while setting the random seed: {e}")

# Example usage:
# set_random_seed(42, use_cuda=True)

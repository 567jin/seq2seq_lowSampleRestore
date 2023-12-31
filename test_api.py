import torch
import numpy as np


def gen_new_tensor(tensor):
    # 随机选择五个索引
    indices = torch.randperm(tensor.size(1))[:5]
    print(indices)
    # 从原始张量中提取选定的五个元素
    selected_elements = tensor[:, indices]

    # 将选定的五个元素变成 64x5 的张量
    new_tensor = selected_elements.view(2, 5)
    return new_tensor


if __name__ == '__main__':
    # tensor = torch.rand(2, 10)
    # for i in range(5):
    #     new_tensor = gen_new_tensor(tensor)
    #     print(tensor, tensor.shape)
    #     print(new_tensor, new_tensor.shape)
    #     print("---------------------------------")
    # tensor = torch.rand(10)
    # print(tensor.size()[0])
    # indices = torch.randperm(tensor.size()[0])[:5]
    # print(indices)
    # selected_elements = tensor[indices]
    # print(tensor, tensor.shape)
    # print(selected_elements, selected_elements.shape)
    x = torch.rand((64, 5))
    print(x.shape)
    print(x.unsqueeze(1).shape)

import torch


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率
    参数：
    optimizer: 优化器
    epoch: 当前训练轮数
    args: 参数配置
    返回值：无
    """

    # 学习率调整方式1  下降的很快 建议每十个epoch调整一次
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # 学习率调整方式2 固定学习率衰减 可以自己设置 但比较吃经验
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    # 根据调整方式更新学习率
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


if __name__ == '__main__':
    class Args:
        def __init__(self, lradj='type1'):
            self.learning_rate = 0.1
            self.lradj = lradj


    args = Args(lradj='type1')
    optimizer = torch.optim.SGD([{'params': [torch.nn.Parameter(torch.randn(100, 100))]}], lr=0.1)
    for epoch in range(10):
        adjust_learning_rate(optimizer, epoch, args)

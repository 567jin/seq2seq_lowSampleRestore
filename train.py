import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import random
from tqdm.auto import tqdm
import mmap
import struct
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import AutoEncoder, Autoencoder_LSTM, ExtraPointsAutoEncoder, ExtraPointsAutoEncoderDeconv
from create_data import My_Dataset, creat_loader


class Args:
    """以类的方式定义参数，模型超参数和一些其他设置"""

    def __init__(self) -> None:
        self.batch_size = 64
        self.lr = 0.001  # 0.001 更好
        self.epochs = 200
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.last_model_name = "out/X_ExtraPointsDeconvLastModel.pth"  # 最后的模型保存路径
        self.best_model_name = "out/X_ExtraPointsDeconvBestModel.pth"  # 最好的模型保存路径

        self.early_stop_epochs = 50  # 验证五次正确率无提升即停止训练
        self.prior = 100  # 先验的loss 只有
        # 随机选择0-300之间的数 按照print_idx==idx打印一下中间结果 这里300表示训练集或验证集最大的批次数
        self.print_idx = np.random.randint(0, 300, 1).item()
        ########################################################################################
        # 修改模型时 一定要修改plot！！！！！！！！#
        self.figPlot_path = r"log\extra_points_autoencoderDeconv_X.svg"
        ########################################################################################


def same_seeds(seed):
    """seed setting"""
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("cuda可用, 并设置相应的随机种子Seed=", seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    torch.backends.cudnn.benchmark = False
    # if True, causes cuDNN to only use deterministic convolution algorithms.
    torch.backends.cudnn.deterministic = True


class Trainer():
    """训练，训练和验证写在一起了"""

    def __init__(self, args, train_loader, val_loader, model):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.train_epochs_loss = []
        self.valid_epochs_loss = []
        self.train_acc = []
        self.val_acc = []

    def train(self):
        min_loss = self.args.prior
        stop_loop = False
        flag = 0
        same_seeds(2023)
        print(self.model)
        parameters = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        # 打印模型参数量
        print('Total parameters: {}'.format(parameters))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        for epoch in range(self.args.epochs):
            if stop_loop:  # 停止训练 start Ploting
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    "--------------------------------------------Training"
                    "------------------------------------------------")
                progress_bar = tqdm(
                    self.train_loader, position=0, leave=True, total=len(self.train_loader))
                self.plot()

            self.model.train()
            train_epoch_loss = []

            # =========================train=======================
            for idx, (x, label) in enumerate(progress_bar):
                optimizer.zero_grad()
                x = x.to(self.args.device, non_blocking=True)
                label = label.to(self.args.device, non_blocking=True)

                out = self.model(x)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

                train_epoch_loss.append(loss.detach().item())

                train_loss = np.average(train_epoch_loss)

                progress_bar.set_description(
                    f"Training   Epoch [{epoch + 1}/{self.args.epochs}]")
                progress_bar.set_postfix(
                    {'loss': train_loss})
                if (epoch + 1) % 10 == 0 and (idx + 1) == self.args.print_idx:
                    print("\n预测值: ", out[:1, :])
                    print("标签值: ", label[:1, :])

            self.train_epochs_loss.append(train_loss)
            # Update learning rate
            print("训练: epoch={}, loss={:.8f}".format(epoch + 1, train_loss))

            # =========================val=========================
            if (epoch + 1) % 2 == 0:  # 每训练两轮验证一次
                val_epoch_loss = []
                print(
                    "---------------------------------------------Valid-----------------------------------------------")
                val_progress_bar = tqdm(
                    self.val_loader, position=0, leave=True, total=len(self.val_loader))
                self.model.eval()
                with torch.no_grad():
                    for idx, (x, label) in enumerate(val_progress_bar):

                        x = x.to(self.args.device, non_blocking=True)
                        label = label.to(self.args.device, non_blocking=True)
                        out = self.model(x)
                        loss = criterion(out, label)

                        val_epoch_loss.append(loss.detach().item())
                        val_loss = np.average(val_epoch_loss)

                        val_progress_bar.set_description(
                            f"Valid   Epoch [{epoch + 1}/{self.args.epochs}]")
                        val_progress_bar.set_postfix(
                            {'loss': val_loss})
                        if (idx + 1) == self.args.print_idx:
                            print("\n预测值: ", out[:1, :])
                            print("标签值: ", label[:1, :])

                self.valid_epochs_loss.append(val_loss)
                # Update learning rate

                if val_loss < min_loss:
                    min_loss = val_loss
                    # 保存验证集上acc最好的模型
                    torch.save(self.model.state_dict(),
                               self.args.best_model_name)
                    print("保存模型... 在epoch={}, loss={:.8f}".format(epoch + 1, val_loss))
                    self.plot()
                else:
                    flag += 1

                if flag >= self.args.early_stop_epochs:
                    print("模型已经没有提升了 终止训练")
                    stop_loop = True
                print(
                    "==============================================End===========================================")
        # =========================save model=====================  训练结束后 保存最后的模型
        print("训练结束，保存最后的模型")
        torch.save(self.model.state_dict(), self.args.last_model_name)
        self.plot()

    def plot(self):
        # =========================plot==========================
        print("\nPloting...")
        plt.figure(figsize=(14, 10))
        plt.subplot(221)
        plt.plot(self.train_epochs_loss[:])
        plt.title("train loss")
        plt.xlabel('epoch')

        plt.subplot(222)
        plt.plot(self.train_epochs_loss, '-o', label="train_loss")
        plt.plot(self.valid_epochs_loss, '-o', label="valid_loss")
        plt.title("epochs loss for train and valid")
        plt.xlabel('epoch')
        plt.legend()

        plt.legend()  # 给图中的线段加标识 即label标签
        plt.savefig(self.args.figPlot_path, format='svg', bbox_inches='tight')
        plt.close()  # 关闭图形 防止打开过多


if __name__ == '__main__':
    args = Args()
    dataset = My_Dataset("data/x.bin", "data/nihex.bin")
    train_loader, val_loader = creat_loader(dataset=dataset, batch_size=args.batch_size)

    # model = AutoEncoder().cuda()
    # model = Autoencoder_LSTM().cuda()
    # model = ExtraPointsAutoEncoder().cuda()
    model = ExtraPointsAutoEncoderDeconv().cuda()
    trainer = Trainer(args, train_loader, val_loader, model)
    trainer.train()
    trainer.plot()

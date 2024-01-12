import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import random
from tqdm.auto import tqdm

from gen_modesl import VAE, loss_vae
from create_data import My_Dataset, creat_loader
from utils.fix_seed import same_seeds


class Args:
    """存储模型参数和配置的类。

    Attributes:
        batch_size (int): 每个训练批次的样本数量。
        lr (float): 学习率。
        epochs (int): 训练周期的数量。
        device (torch.device): 模型训练使用的设备 (CPU 或 GPU)。
        last_model_name (str): 最终模型的保存路径。
        best_model_name (str): 最佳模型的保存路径。
        early_stop_epochs (int): 当验证正确率连续提升次数达到此值时停止训练。
        prior (int): 先验损失的阈值。
        print_idx (int): 随机选择的打印中间结果的批次索引。
        figPlot_path (str): 保存图表的文件路径。
    """

    def __init__(self) -> None:
        self.batch_size = 64
        self.lr: float = 0.001  # 学习率，0.001 为推荐值
        self.epochs: int = 200
        self.device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.last_model_name: str = "out/X_LinearLastModel.pth"  # 最后的模型保存路径
        self.best_model_name: str = "out/X_LinearBestModel.pth"  # 最佳模型的保存路径
        self.early_stop_epochs: int = 50  # 当验证正确率连续提升超过此值时，停止训练
        self.prior: int = 100  # 先验损失阈值
        # 随机选择 0-300 之间的数，按 print_idx==idx 打印中间结果，300 表示训练集或验证集最大的批次数
        self.print_idx: int = np.random.randint(0, 300)
        # 当修改模型时，一定要修改 plot
        self.figPlot_path: str = r"log\LinearAutoencoder_X.svg"


class Trainer:
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
                # 刷新进度条
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

                x_hat, z_mean, z_logvar = self.model(x)
                loss = loss_vae(x_hat, label, z_mean, z_logvar, criterion)

                loss.backward()
                optimizer.step()

                train_epoch_loss.append(loss.detach().item())
                train_loss = np.average(train_epoch_loss)

                progress_bar.set_description(
                    f"Training   Epoch [{epoch + 1}/{self.args.epochs}]")
                progress_bar.set_postfix(
                    {'loss': train_loss})
                if (epoch + 1) % 10 == 0 and (idx + 1) == self.args.print_idx:
                    print("\n预测值: ", x_hat[:1, :])
                    print("标签值: ", label[:1, :])
            # TODO: 在线性自编码器中，对于输入10输出10的二阶拟合任务，效果很好，loss下降太快，因此在每个epoch才记录loss 导致画图很难看 可以改成在每次迭代记录loss
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
                        x_hat, z_mean, z_logvar = self.model(x)
                        loss = loss_vae(x_hat, label, z_mean, z_logvar, criterion)

                        val_epoch_loss.append(loss.detach().item())
                        val_loss = np.average(val_epoch_loss)

                        val_progress_bar.set_description(
                            f"Valid   Epoch [{epoch + 1}/{self.args.epochs}]")
                        val_progress_bar.set_postfix(
                            {'loss': val_loss})
                        if (idx + 1) == self.args.print_idx:
                            print("\n预测值: ", x_hat[:1, :])
                            print("标签值: ", label[:1, :])

                self.valid_epochs_loss.append(val_loss)
                # 关闭进度条
                val_progress_bar.close()
                if val_loss < min_loss:
                    flag = 0
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
        progress_bar.close()  # 关闭进度条
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
        # plt.plot(self.train_epochs_loss, '-o', label="train_loss")
        plt.plot(self.valid_epochs_loss, '-o', label="valid_loss")
        plt.title("epochs loss for valid")
        plt.xlabel('epoch')
        plt.legend()

        plt.legend()  # 给图中的线段加标识 即label标签
        plt.savefig(self.args.figPlot_path, format='svg', bbox_inches='tight')
        plt.close()  # 保存完之后就关闭图形 防止打开过多


if __name__ == '__main__':
    args = Args()
    dataset = My_Dataset(sig_filename="data/x.bin", label_filename="data/nihex.bin", extra_points=True)
    train_loader, val_loader = creat_loader(dataset=dataset, batch_size=args.batch_size)

    model = VAE(input_dim=5, output_dim=10, latent_dim=2).cuda()
    trainer = Trainer(args, train_loader, val_loader, model)
    trainer.train()
    trainer.plot()

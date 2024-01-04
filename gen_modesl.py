from torch import nn
import torch
from create_data import My_Dataset, creat_loader
import math
from torch.autograd import Variable


# 定义变分自编码器的编码器
class LinearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mean = nn.Linear(32, latent_dim)
        self.fc3_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        z_mean = self.fc3_mean(x)
        z_logvar = self.fc3_logvar(x)
        return z_mean, z_logvar


# 定义变分自编码器的解码器
class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(z))
        return x_hat


# 定义完整的 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = LinearEncoder(input_dim, latent_dim)
        self.decoder = LinearDecoder(latent_dim, output_dim)

    def reparameterize(self, z_mean, z_logvar):
        epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        # encoder 两个输入合在一起当做decoder的输入z
        z = self.reparameterize(z_mean, z_logvar)
        # decoder的输出就是预测的重建值x
        x_hat = self.decoder(z)
        return x_hat, z_mean, z_logvar


def loss_vae(recon_x, x, z_mean, z_logvar, criterion):
    """
    计算 VAE 的总损失，包括重建损失和 KL 散度损失
    recon_x: 生成的 x
    x: 原始的 x
    z_mean: 潜在空间均值
    z_logvar: 潜在空间方差的对数
    criterion: 用于重建损失的损失函数
    """
    # 重建损失
    reconstruction_loss = criterion(recon_x, x)
    # KL 散度损失
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    # 总损失
    total_loss = reconstruction_loss + kl_loss
    return total_loss


if __name__ == '__main__':
    model = VAE(input_dim=5, output_dim=10, latent_dim=2)
    dataset = My_Dataset(sig_filename=r"data\x.bin", label_filename=r"data\nihex.bin", extra_points=True)
    train_loader, val_loader = creat_loader(dataset=dataset, batch_size=1)
    x, y = next(iter(train_loader))

    x_hat, z_mean, z_logvar = model(x)
    print("x= ", x, x.shape)
    print("y= ", y, y.shape)
    print("out= ", x_hat, x_hat.shape)

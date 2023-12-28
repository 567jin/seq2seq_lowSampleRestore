from torch import nn
import torch
from create_data import My_Dataset, creat_loader
import math


class Encoder(nn.Module):
    def __init__(self, p=0):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(100, 300),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(300, 100),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(100, 5)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, p=0):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(5, 100),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(100, 300),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(300, 100),
            nn.ReLU(True),
            # nn.Dropout(p),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    """线性自编码器的模型，编码器和解码器都是线性层 """

    def __init__(self, p=0):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(p=p)
        self.decoder = Decoder(p=p)

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x


class Autoencoder_LSTM(nn.Module):
    def __init__(self):
        super(Autoencoder_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=32, num_layers=5, batch_first=True)
        self.decoder = nn.LSTM(input_size=32, hidden_size=1, num_layers=5, batch_first=True)

    def forward(self, x):
        # LSTM的输入需要加一个维度
        x = x.unsqueeze(2)
        encoded_outputs, _ = self.encoder(x)
        decoded_outputs, _ = self.decoder(encoded_outputs)
        return decoded_outputs.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TTED(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, num_heads=2, num_layers=2):
        super(TTED, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),  # 64*10*64   64*10*256
            nn.ReLU(),
            PositionalEncoding(d_model),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead=num_heads),
                num_layers=num_layers
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            PositionalEncoding(d_model),
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model, nhead=num_heads),
                num_layers=num_layers
            )
        )

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x_enc = self.encoder(x)
        print(x_enc.shape)  # 64*10*256
        x_dec = self.decoder(x_enc)
        output = self.fc(x_dec)
        return output


if __name__ == '__main__':
    # dataset = My_Dataset(r"data\x.bin", r"data\nihex.bin")
    # train_loader, val_loader = creat_loader(dataset=dataset, batch_size=64)
    # x = torch.rand((64, 10), dtype=torch.float32).unsqueeze(2)
    x = torch.rand((64, 10, 64), dtype=torch.float32)
    # x, y = next(iter(train_loader))
    # model = Autoencoder_LSTM().cuda()
    model = TTED(input_dim=64, output_dim=10).cuda()
    print(model)
    out = model(x.cuda())
    # print(y, y.shape)
    print(out[:1, :], out.shape)
    print(x[:1, :])

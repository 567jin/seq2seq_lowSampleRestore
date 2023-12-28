from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np


class My_Dataset(Dataset):
    def __init__(self, sig_filename, label_filename) -> None:
        with open(sig_filename, 'rb') as f:
            x_data = torch.from_numpy(np.fromfile(f, dtype=np.float64))
            x = x_data

        self.X = []
        for idx, i in enumerate(range(0, len(x), 10)):
            self.X.append(x[i:i + 10].to(dtype=torch.float32))

        with open(label_filename, 'rb') as f:
            y_data = torch.from_numpy(np.fromfile(f, dtype=np.float64))
            y = y_data

        self.Y = []
        for idx, i in enumerate(range(0, len(y), 10)):
            self.Y.append(y[i:i + 10].to(dtype=torch.float32))

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.X)


def creat_loader(dataset, batch_size=1):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    print("构建数据集...")
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, prefetch_factor=2, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, prefetch_factor=2, drop_last=False)
    print("数据集构建成功...")
    return train_loader, val_loader


if __name__ == '__main__':
    dataset = My_Dataset(r"data\x.bin", r"data\nihex.bin")
    print(len(dataset))
    train_loader, val_loader = creat_loader(dataset=dataset)

    print(train_loader)
    print(val_loader)
    data, target = next(iter(train_loader))  # type:torch.Tensor
    print(type(data))

    assert isinstance(data, torch.Tensor)
    print(data.shape)  # 64*10
    print(target.shape)  # 64*10
    print(data)
    print(target)

import torch
import numpy as np
from models import AutoEncoder, ExtraPointsAutoEncoder, ExtraPointsAutoEncoderDeconv
from gen_models import VAE
from create_data import creat_loader, My_Dataset
from plot import plot_linear


def predict(model, x):
    model.eval()
    with torch.no_grad():
        out = model(x)

    return out


def save_val(output_file, model, X, length):
    all_predictions = np.empty((length, 10), dtype=np.float64)
    for i in range(length):
        x = X[i]
        output = predict(model, x)

        # 将输出张量转换为 NumPy 数组
        # output_np = output.numpy()
        all_predictions[i] = output.numpy()

    # 保存输出到文件
    # np.save(output_file, all_predictions, allow_pickle=False)
    all_predictions.tofile(output_file)


if __name__ == '__main__':
    # model = AutoEncoder()
    # model = ExtraPointsAutoEncoderDeconv()
    # model = ExtraPointsAutoEncoder()
    model = VAE(input_dim=5, output_dim=10, latent_dim=2)
    model.load_state_dict(torch.load("out/X_gen_VAE_BestModel.pth"))
    dataset = My_Dataset("data/x.bin", label_filename="data/nihex.bin", extra_points=True)
    X = dataset.X
    Y = dataset.Y
    x = X[0]
    y = Y[0]
    print(model)
    data_length = len(X)
    print("数据长度: ", data_length)
    print(x.shape)
    # out = predict(model, x.unsqueeze(0))  # 卷积的输入需要是 1,5
    out, z_mean, z_logvar = model(x)  # 线性VAE
    # out= predict(model, x)
    print("标签: ", y)
    print("模型输出: ", out)
    print(out.shape)
    plot_linear(out.detach().cpu().numpy(), y.detach().cpu().numpy(), title="VAE-based Model")
    # save_val(output_file="output_p2.bin", model=model, X=X, length=data_length)

import matplotlib.pyplot as plt
import numpy as np


def plot_linear(predict, label):
    plt.figure(figsize=(12, 4))
    x = np.arange(10)
    plt.plot(x, predict, label="predict", linewidth=2, color="red")
    plt.plot(x, label, label="label", linewidth=2, color="blue")
    plt.title('amplitude')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sig = np.fromfile('output_p.bin', dtype=np.float64)
    sig2 = np.fromfile('data/p_2.bin', dtype=np.float64)
    plot_linear(sig[0:10], sig2[0:10])
    print(len(sig))
    print(len(sig2))

import os
import torch
from torchvision import datasets, transforms

# 1. 下载MNIST数据集
def download_mnist_data(data_dir='./data'):
    # 创建data目录（如果不存在的话）
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 设置下载的transform
    download_transform = transforms.Compose([transforms.ToTensor()])
    
    # 下载数据集
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=download_transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=download_transform)
    
    print("数据集下载完成!")
    return train_data, test_data

# 2. 读取MNIST数据集
def load_mnist_data(data_dir='./data'):
    # 加载训练集和测试集
    train_data = datasets.MNIST(root=data_dir, train=True, download=False, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=False, transform=transforms.ToTensor())
    
    return train_data, test_data

# 3. 输出MNIST数据集的信息
def print_mnist_data_info(train_data, test_data):
    print("训练集样本数:", len(train_data))
    print("测试集样本数:", len(test_data))
    print("训练集每个样本的维度:", train_data[0][0].shape)  # 输出一个样本的维度
    print("测试集每个样本的维度:", test_data[0][0].shape)

# 运行代码
if __name__ == "__main__":
    # 下载数据集并读取
    train_data, test_data = download_mnist_data()

    # 打印数据集信息
    print_mnist_data_info(train_data, test_data)

import torch
import torchvision.datasets as dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from CNN import CNN

batch_size = 64  # 分批训练数据、每批数据量
learning_rate = 1e-2  # 学习率
num_epoches = 5  # 训练次数
DOWNLOAD_MNIST = True  # 是否网上下载数据

# Mnist digits dataset
if not (os.path.exists('mnist')) or not os.listdir('mnist'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = dataset.MNIST(root="mnist",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=DOWNLOAD_MNIST)

test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=DOWNLOAD_MNIST)
# DataLoader
# dataset:包含所有数据的数据集
# batch_size:每一小组所包含数据的数量
# Shuffle : 是否打乱数据位置，当为Ture时打乱数据
# num_workers : 使用线程的数量，当为0时数据直接加载到主程序，默认为0
# drop_last:布尔类型，为T时将会把最后不足batch_size的数据丢掉，为F将会把剩余的数据作为最后一小组
# timeout：默认为0。当为正数的时候，这个数值为时间上限，每次取一个batch超过这个值的时候会报错。此参数必须为正数。
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(dataloader)返回的是一个迭代器，
# 然后可以使用next()访问。
# 也可以使用enumerate(dataloader)的形式访问。
# images, labels = next(iter(train_loader))
# print(labels)
# 拼接图像 padding为各边距
# img = torchvision.utils.make_grid(images,padding=10)
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# cv2.imshow('win', img)
# key_pressed = cv2.waitKey(0)

# CNN
cnn = CNN()
# LOSS
loss_func = torch.nn.CrossEntropyLoss()
# OPTIMIZER
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# TRAINING
for epoch in range(num_epoches):
    for itr, (image, lable) in enumerate(train_loader):
        output = cnn(image)
        loss = loss_func(output, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch is {}, ite is "
          "{}/{}, loss is {}".format(epoch + 1, itr,
                                     len(train_data) // batch_size,
                                     loss.item()))
    # EVAL/TEST
    loss_test = 0
    accuracy = 0
    for itr, (image, lable) in enumerate(test_loader):
        output = cnn(image)
        loss_test += loss_func(output, lable)
        # 所在行 _:value pred:index
        _, pred = output.max(1)
        # (pred==lable).sum()返回张量
        accuracy += (pred == lable).sum().item()
        print("itr is {},output is {}".format(itr + 1, output))
    accuracy = accuracy / (len(test_data))
    loss_test = loss_test / (len(test_data) // batch_size)
    print("epoch is {}, accuracy is {},loss test is {}"
          .format(epoch + 1, accuracy, loss_test.item()))

# SAVE
torch.save(cnn, "model/mnist_model.pkl")

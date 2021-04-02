import torch
from matplotlib import pyplot as plt
import numpy as np
import random

# print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size = 32
# 随机种子
seed = torch.manual_seed(10)
features = torch.randn(num_examples, num_inputs)
# 矢量乘
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# print(labels[:20])
# 添加一些误差
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# print(labels[:20])


# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        # print(indices[i: min(i + batch_size, num_examples)])
        # index_select(0,j) 行维度选择idx为j
        yield features.index_select(0, j), labels.index_select(0, j)


# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 学习率和epoch
lr = 0.01
num_epochs = 10
# 随机初始化模型
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
b = torch.zeros(1)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 线性模型
def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    return torch.mm(X, w) + b


# 损失函数
def squared_loss(y_hat, y):  # 本函数已保存在pytorch_d2lzh包中方便以后使用
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size


net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    pred = torch.mm(X, w) + b
    print('pred:{},y:{}'.format(pred, y))

print(w)
print(b)
# print(features[:1], labels[:1])

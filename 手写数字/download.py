import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import cv2

test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

cnn = torch.load("model/mnist_model.pkl")

# loss
# eval/test
loss_test = 0
accuracy = 0

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
for itr, (images, lables) in enumerate(test_loader):
    output = cnn(images)
    # 所在行 _:value pred:index
    _, pred = output.max(1)
    # (pred==lable).sum()返回张量
    accuracy += (pred == lables).sum().item()

    images = images.numpy()
    lables = lables.numpy()
    pred = pred.numpy()
    # print(image.shape[0])

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = lables[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)
        print("label", im_label)
        print("pred", im_pred)
        cv2.imshow("imdata", im_data)
        cv2.waitKey(10)

accuracy = accuracy / len(test_loader)
print(accuracy)

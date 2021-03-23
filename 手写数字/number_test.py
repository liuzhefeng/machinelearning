import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset

train_data = dataset.MNIST(root="mnist",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_data = dataset.MNIST(root="mnist",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)




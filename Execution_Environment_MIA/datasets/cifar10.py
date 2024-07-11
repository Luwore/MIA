import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    x = train_dataset.data
    y = np.array(train_dataset.targets)
    test_x = test_dataset.data
    test_y = np.array(test_dataset.targets)

    return (x, y), (test_x, test_y)

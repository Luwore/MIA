import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

from torchvision.models import resnet18
import torch.optim as optim
import torch.nn as nn

MODEL_PATH = './model/'
DATA_PATH = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

criterion = nn.CrossEntropyLoss()


def get_resnet18():
    model = resnet18(weights=None, num_classes=10).to(device)
    return model


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Training loss: {running_loss / len(train_loader)}')


def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy}')
    print(classification_report(all_labels, all_preds))


def get_data_indices(data_size, target_train_size=10000, n_shadow=10):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    target_data_indices = indices[:target_train_size]
    shadow_data_indices = np.array_split(indices[target_train_size:], n_shadow)
    return target_data_indices, shadow_data_indices


def load_data(data_name):
    file_path = DATA_PATH + data_name
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    with np.load(file_path) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    print(f'Loaded {data_name}:')
    print(f'train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
    print(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')
    return train_x, train_y, test_x, test_y


def train_target_model(args):
    dataset = load_data('target_data.npz')
    train_x, train_y, test_x, test_y = dataset
    train_x = train_x.transpose((0, 3, 1, 2))
    test_x = test_x.transpose((0, 3, 1, 2))

    train_loader = DataLoader(TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_y).long()),
                              batch_size=args.target_batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).long()),
                             batch_size=args.target_batch_size, shuffle=False)

    model = get_resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.target_learning_rate, momentum=args.target_momentum,
                          weight_decay=args.target_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.target_epochs)

    for epoch in range(args.target_epochs):
        print(f'Target Model - Epoch {epoch + 1}/{args.target_epochs}')
        train(model, train_loader, criterion, optimizer, device)
        test(model, test_loader, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), MODEL_PATH + 'target_model.pth')
        print(f'Target model saved to {MODEL_PATH}target_model.pth')
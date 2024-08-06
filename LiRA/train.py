import functools
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from absl import app, flags

from config import FLAGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Ensure the image is in the correct format and shape
        if isinstance(image, torch.Tensor):
            if image.shape[1] != 32 or image.shape[2] != 32:  # Check if the shape is not (C, H, W)
                image = image.permute(1, 2, 0)  # Convert to (H, W, C)
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)
            # Debug: Print the transformed size of the image

        return image, label


def augment(shift: int):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=shift),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    return transform


class TrainLoop:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, data_loader, epoch, log_interval=100):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                      f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        return total_loss / len(data_loader.dataset)

    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
              f'({accuracy:.0f}%)\n')
        return test_loss, accuracy


def network(arch: str):
    if arch == 'cnn32-3-max':
        return SimpleCNN(num_classes=10)
    elif arch == 'wrn28-2':
        return WideResNet(depth=28, width_factor=2, num_classes=10)
    raise ValueError('Architecture not recognized', arch)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, depth, width_factor, num_classes):
        super(WideResNet, self).__init__()
        self.depth = depth
        self.width_factor = width_factor
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16, 16 * self.width_factor, self.depth // 3, stride=1)
        self.layer2 = self._make_layer(16 * self.width_factor, 32 * self.width_factor, self.depth // 3, stride=2)
        self.layer3 = self._make_layer(32 * self.width_factor, 64 * self.width_factor, self.depth // 3, stride=2)
        self.bn = nn.BatchNorm2d(64 * self.width_factor)
        self.fc = nn.Linear(64 * self.width_factor, self.num_classes)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = []
        strides = [stride] + [1] * (blocks - 1)
        for stride in strides:
            layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, x.size()[3])  # Pool to the size of the feature map
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_data(seed):
    DATA_DIR = os.path.join(os.environ['HOME'], 'pytorch_data')

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    if os.path.exists(os.path.join(FLAGS.logdir, "x_train.npy")):
        inputs = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))
        labels = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))
    else:
        print("First time, creating dataset")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)

        inputs = np.array([np.array(img) for img, _ in full_train_dataset])
        labels = np.array([label for _, label in full_train_dataset])

        # Normalize inputs to [-1, 1]
        inputs = (inputs / 127.5) - 1
        np.save(os.path.join(FLAGS.logdir, "x_train.npy"), inputs)
        np.save(os.path.join(FLAGS.logdir, "y_train.npy"), labels)

    nclass = np.max(labels) + 1

    np.random.seed(seed)
    if FLAGS.num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(axis=0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    if FLAGS.only_subset is not None:
        keep[FLAGS.only_subset:] = 0

    xs = inputs[keep]
    ys = labels[keep]

    if FLAGS.augment == 'weak':
        aug_transform = augment(4)
    elif FLAGS.augment == 'mirror':
        aug_transform = augment(0)
    elif FLAGS.augment == 'none':
        aug_transform = transforms.Compose([transforms.ToTensor()])
    else:
        raise ValueError("Unknown augmentation type")

    # Convert numpy arrays back to a dataset
    dataset = [(torch.tensor(img).permute(2, 0, 1), torch.tensor(label)) for img, label in zip(xs, ys)]

    # Applying augmentation
    train_dataset = AugmentedDataset(dataset, transform=aug_transform)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch, shuffle=True, num_workers=2)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch, shuffle=False, num_workers=2)

    return train_loader, test_loader, xs, ys, keep, nclass


def main(argv):
    del argv
    seed = FLAGS.seed
    if seed is None:
        import time
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())

    torch.manual_seed(seed)

    model = network(FLAGS.arch).to(device)

    print(f'Model parameters: {list(model.parameters())}')  # Debug statement
    print(f'Experiment number {FLAGS.expid} of {FLAGS.num_experiments}')

    optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader, xs, ys, keep, nclass = get_data(seed)
    train_loop = TrainLoop(model, optimizer, criterion, device)

    logdir = os.path.join(FLAGS.logdir, f'experiment_{FLAGS.expid}_of_{FLAGS.num_experiments}')
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    # Create 'ckpt' directory
    checkpoint_dir = os.path.join(logdir, 'ckpt')
    os.makedirs(checkpoint_dir)

    writer = SummaryWriter(log_dir=logdir)

    best_acc = 0
    best_acc_epoch = -1

    for epoch in range(1, FLAGS.epochs + 1):
        train_loss = train_loop.train_step(train_loader, epoch)
        test_loss, accuracy = train_loop.evaluate(test_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        if accuracy > best_acc:
            best_acc = accuracy
            best_acc_epoch = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))

        if epoch % FLAGS.save_steps == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))

        if FLAGS.patience and epoch - best_acc_epoch > FLAGS.patience:
            print("Early stopping!")
            break

    # Save hyperparameters and 'keep' array
    hparams = {key: value.value if hasattr(value, 'value') else value for key, value in FLAGS.flag_values_dict().items()}
    with open(os.path.join(logdir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f)
    np.save(os.path.join(logdir, 'keep.npy'), keep)

    writer.close()


if __name__ == '__main__':
    app.run(main)

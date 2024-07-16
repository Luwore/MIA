import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import json
import numpy as np
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
epochs = 200
learning_rate = 0.1
num_models = 16
save_path = 'exp/cifar10/'

# Prepare CIFAR-10 data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# Training function
def train_model(model, criterion, optimizer, num_epochs=epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')


# Main function to train and save models
def main():
    os.makedirs(save_path, exist_ok=True)

    for i in range(num_models):
        print(f'Training model {i + 1}/{num_models}')
        model = resnet18(weights=None, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        train_model(model, criterion, optimizer)

        # Save model
        model_dir = os.path.join(save_path, f'experiment_{i + 1}_of_{num_models}')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'ckpt.pth'))
        with open(os.path.join(model_dir, 'hparams.json'), 'w') as f:
            json.dump({'lr': learning_rate, 'epochs': epochs}, f)
        np.save(os.path.join(model_dir, 'keep.npy'), np.array([1]))


if __name__ == '__main__':
    main()

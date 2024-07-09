import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from torchvision.models import resnet18
import torch.optim as optim
import torch.nn as nn
from utility import load_data

MODEL_PATH = './model/'
DATA_PATH = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_target_model(args, dataset=load_data('target_data.npz')):
    train_x, train_y, test_x, test_y = dataset

    train_loader = DataLoader(TensorDataset(torch.Tensor(train_x).permute(0, 3, 1, 2),
                                            torch.Tensor(train_y).long()),
                              batch_size=args.target_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(TensorDataset(torch.Tensor(test_x).permute(0, 3, 1, 2),
                                           torch.Tensor(test_y).long()),
                             batch_size=args.target_batch_size, shuffle=False, num_workers=2)

    model = get_resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.target_learning_rate, momentum=args.target_momentum,
                          weight_decay=args.target_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.target_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.target_epochs):
        print(f'Target Model - Epoch {epoch + 1}/{args.target_epochs}')
        train(model, train_loader, criterion, optimizer, device)
        test(model, test_loader, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), MODEL_PATH + 'target_model.pth')
        print(f'Target model saved to {MODEL_PATH}target_model.pth')

    attack_x, attack_y = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            attack_x.append(outputs.cpu().numpy())
            attack_y.append(np.ones(inputs.size(0)))

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            attack_x.append(outputs.cpu().numpy())
            attack_y.append(np.zeros(inputs.size(0)))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y).astype('int32')

    np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
    return attack_x, attack_y, np.concatenate([train_y, test_y])

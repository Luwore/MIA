import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

from train_victim_model import get_resnet18, train, test, load_data, criterion

MODEL_PATH = './model/'
DATA_PATH = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_shadow_models(args):
    attack_x, attack_y = [], []
    classes = []

    for i in range(args.n_shadow):
        print(f'Training shadow model {i}')
        data = load_data(f'shadow{i}_data.npz')
        train_x, train_y, test_x, test_y = data

        train_x = train_x.reshape((-1, 32, 32, 3))
        test_x = test_x.reshape((-1, 32, 32, 3))
        train_y = train_y.flatten()
        test_y = test_y.flatten()

        if train_x.ndim == 4 and train_x.shape[-1] == 3:
            train_x = train_x.transpose((0, 3, 1, 2)).astype(np.float32)
        if test_x.ndim == 4 and test_x.shape[-1] == 3:
            test_x = test_x.transpose((0, 3, 1, 2)).astype(np.float32)

        train_loader = DataLoader(TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_y).long()),
                                  batch_size=args.target_batch_size, shuffle=True)
        if len(test_x) > 0 and len(test_y) > 0:
            test_loader = DataLoader(TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).long()),
                                     batch_size=args.target_batch_size, shuffle=False)
        else:
            test_loader = None

        model = get_resnet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.target_learning_rate, momentum=args.target_momentum,
                              weight_decay=args.target_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.target_epochs)

        for epoch in range(args.target_epochs):
            print(f'Shadow Model {i + 1} - Epoch {epoch + 1}/{args.target_epochs}')
            train(model, train_loader, criterion, optimizer, device)
            if test_loader:
                test(model, test_loader, device)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), MODEL_PATH + f'shadow_model_{i}.pth')

        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                attack_x.append(outputs.cpu().numpy())
                attack_y.append(np.ones(inputs.size(0)))

            if test_loader:
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    attack_x.append(outputs.cpu().numpy())
                    attack_y.append(np.zeros(inputs.size(0)))

        classes.append(np.concatenate([train_y, test_y]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y).astype('int32')
    classes = np.concatenate(classes)

    if args.save_model:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes

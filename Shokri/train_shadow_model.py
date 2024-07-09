import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from train_victim_model import get_resnet18, train, test, load_data
from main import logger

MODEL_PATH = './model/'
DATA_PATH = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_shadow_models(args):
    attack_x, attack_y = [], []
    classes = []

    for i in range(args.n_shadow):
        logger.info(f'Training shadow model {i}')
        dataset = load_data(f'shadow{i}_data.npz')
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
            logger.info(f'Shadow Model {i + 1} - Epoch {epoch + 1}/{args.target_epochs}')
            train(model, train_loader, criterion, optimizer, device)
            if test_loader:
                test(model, test_loader, device)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), MODEL_PATH + f'shadow_model_{i}.pth')
            logger.info(f'Shadow model {i} saved to {MODEL_PATH}shadow_model_{i}.pth')

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

    np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)
    return attack_x, attack_y, classes

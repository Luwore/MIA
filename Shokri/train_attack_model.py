import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from train_victim_model import get_resnet18, train
from main import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_attack_data():
    fname = './model/attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = './model/attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_attack_model(args):
    dataset = load_attack_data()
    train_x, train_y, test_x, test_y = dataset

    if train_x.ndim == 2 and train_x.shape[1] == 10:
        train_x = train_x.reshape((-1, 1, 10, 10))
    if test_x.ndim == 2 and test_x.shape[1] == 10:
        test_x = test_x.reshape((-1, 1, 10, 10))

    train_y = train_y.flatten()
    test_y = test_y.flatten()

    train_x = np.repeat(train_x, 3, axis=1)
    test_x = np.repeat(test_x, 3, axis=1)

    train_classes = np.load('./model/attack_train_classes.npz')['arr_0']
    test_classes = np.load('./model/attack_test_classes.npz')['arr_0']

    if len(train_classes) != len(train_x):
        train_classes = np.tile(train_classes, len(train_x) // len(train_classes) + 1)[:len(train_x)]
    if len(test_classes) != len(test_x):
        test_classes = np.tile(test_classes, len(test_x) // len(test_classes) + 1)[:len(test_x)]

    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []

    for c in unique_classes:
        logger.info(f'Training attack model for class {c}...')
        c_train_indices = train_indices[train_classes == c]
        if len(c_train_indices) == 0:
            logger.info(f'No training samples for class {c}. Skipping...')
            continue
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]

        c_test_indices = test_indices[test_classes == c]
        if len(c_test_indices) == 0:
            logger.info(f'No test samples for class {c}. Skipping...')
            continue
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        c_model = get_resnet18().to(device)
        optimizer = torch.optim.Adam(c_model.parameters(), lr=args.attack_learning_rate)
        criterion = nn.CrossEntropyLoss()
        c_train_loader = DataLoader(TensorDataset(torch.tensor(c_train_x).float(), torch.tensor(c_train_y).long()),
                                    batch_size=args.attack_batch_size, shuffle=True)
        for epoch in range(args.attack_epochs):
            logger.info(f'Attack Model for class {c} - Epoch {epoch + 1}/{args.attack_epochs}')
            train(c_model, c_train_loader, criterion, optimizer, device)

        c_model.eval()
        with torch.no_grad():
            c_test_loader = DataLoader(TensorDataset(torch.tensor(c_test_x).float(), torch.tensor(c_test_y).long()),
                                       batch_size=args.attack_batch_size, shuffle=False)
            c_preds = []
            for inputs, labels in c_test_loader:
                inputs = inputs.to(device)
                outputs = c_model(inputs)
                _, preds = torch.max(outputs, 1)
                c_preds.extend(preds.cpu().numpy())

            true_y.append(c_test_y)
            pred_y.append(c_preds)

    if true_y and pred_y:
        true_y = np.concatenate(true_y)
        pred_y = np.concatenate(pred_y)
        logger.info('Testing Accuracy:', accuracy_score(true_y, pred_y))
        logger.info(classification_report(true_y, pred_y))
    else:
        logger.warn('No data to evaluate.')

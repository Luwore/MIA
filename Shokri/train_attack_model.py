import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from main import logger
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def get_nn_model(n_in, n_hidden, n_out):
    model = SimpleNN(n_in, n_hidden, n_out)
    return model


def load_attack_data():
    train_file = './model/attack_train_data.npz'
    test_file = './model/attack_test_data.npz'
    test_classes = np.load('./model/attack_test_classes.npz')
    train_classes = np.load('./model/attack_train_classes.npz')

    with np.load(train_file) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    with np.load(test_file) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    return (train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32'),
            test_classes, train_classes)


def train_model(c_dataset, args):
    c_train_x, c_train_y, c_test_x, c_test_y = c_dataset

    # Flatten the input data if it's not already flattened
    c_train_x = c_train_x.reshape(c_train_x.shape[0], -1)
    c_test_x = c_test_x.reshape(c_test_x.shape[0], -1)

    # Create DataLoader for training and testing
    c_train_loader = DataLoader(TensorDataset(torch.tensor(c_train_x).float(), torch.tensor(c_train_y).long()),
                                batch_size=args.attack_batch_size, shuffle=True)
    c_test_loader = DataLoader(TensorDataset(torch.tensor(c_test_x).float(), torch.tensor(c_test_y).long()),
                               batch_size=args.attack_batch_size, shuffle=False)

    input_size = c_train_x.shape[1]
    hidden_size = 128  # You can adjust this size
    output_size = len(np.unique(c_train_y))  # Assuming the classes are labeled 0 to num_classes-1

    # Initialize model, loss function, and optimizer
    model = get_nn_model(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.attack_learning_rate)

    # Training loop
    for epoch in range(args.attack_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in c_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f'Epoch [{epoch + 1}/{args.attack_epochs}], Loss: {running_loss / len(c_train_loader):.4f}')

    # Evaluation
    model.eval()
    c_preds = []
    with torch.no_grad():
        for inputs, labels in c_test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            c_preds.extend(preds.cpu().numpy())

    return np.array(c_preds)


def train_attack_model(args):
    train_x, train_y, test_x, test_y, test_classes, train_classes = load_attack_data()

    train_y = train_y.flatten()
    test_y = test_y.flatten()

    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes['arr_0'])

    true_y = []
    pred_y = []

    for c in unique_classes:
        logger.info(f'Training attack model for class {c}...')
        c_train_indices = train_indices[train_classes['arr_0'] == c]
        if len(c_train_indices) == 0:
            logger.info(f'No training samples for class {c}. Skipping...')
            continue
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]

        c_test_indices = test_indices[test_classes['arr_0'] == c]
        if len(c_test_indices) == 0:
            logger.info(f'No test samples for class {c}. Skipping...')
            continue
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)

        c_pred_y = train_model(c_dataset, args)

        true_y.append(c_test_y)
        pred_y.append(c_pred_y)

    logger.info('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    logger.info('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    logger.info(classification_report(true_y, pred_y, zero_division=0))
    logger.info('Attack model training completed.')

import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from Execution_Environment_MIA.models.SimpleNN import get_nn_model
from Execution_Environment_MIA.models.resnet import get_resnet18
from Execution_Environment_MIA.utils import device, MODEL_PATH, collect_test_data, train, test

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s \n %(message)s',
    handlers=[
        logging.FileHandler("log.log"),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

logger = logging.getLogger(__name__)


def perform_attack(args):
    logger.info('-' * 10 + 'TRAIN TARGET' + '-' * 10)
    train_target_model(args)

    logger.info('-' * 10 + 'TRAIN SHADOW' + '-' * 10)
    train_shadow_models(args)

    logger.info('-' * 10 + 'TRAIN ATTACK' + '-' * 10)
    train_attack_model(args)


def train_target_model(args):
    train_x, train_y, test_x, test_y = get_data('target')

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
        logger.info(f'Target Model - Epoch {epoch + 1}/{args.target_epochs}')
        train(model, train_loader, criterion, optimizer, device)
        test(model, test_loader, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), MODEL_PATH + 'target_model.pth')
        logger.info(f'Target model saved to {MODEL_PATH}target_model.pth')

    return collect_test_data(model, train_loader, test_loader, train_y, test_y, 'attack_test_data.npz')


def train_shadow_models(args):
    attack_x, attack_y = [], []
    classes = []

    for i in range(args.n_shadow):
        logger.info(f'Training shadow model {i}')
        dataset = get_data(f'shadow{i}')
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
    np.savez(MODEL_PATH + 'attack_train_classes.npz', classes)
    return attack_x, attack_y, classes


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
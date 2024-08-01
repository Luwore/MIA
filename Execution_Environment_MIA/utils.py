import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_PATH + '/source/model/'
DATA_PATH = BASE_PATH + '/source/data/'

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


def get_data_indices(data_size, target_train_size, n_shadow):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    target_data_indices = indices[:target_train_size]
    shadow_data_indices = np.array_split(indices[target_train_size:], n_shadow)
    return target_data_indices, shadow_data_indices


def collect_test_data(model, train_loader, test_loader, train_y, test_y, name):
    logger.debug(f'Collecting test data for attack model')
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
    classes = np.concatenate([train_y, test_y])

    np.savez(MODEL_PATH + name, attack_x, attack_y)
    np.savez(MODEL_PATH + 'attack_test_classes.npz', classes)
    logger.info(f'Attack data saved to {MODEL_PATH}{name}')
    return attack_x, attack_y, classes


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

    logger.info(f'Training loss: {running_loss / len(train_loader)}')


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
    logger.info(f'Test Accuracy: {accuracy}')
    logger.info(classification_report(all_labels, all_preds))


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

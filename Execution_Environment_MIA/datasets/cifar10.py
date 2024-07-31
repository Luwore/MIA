import os
from dataclasses import dataclass

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np

from Execution_Environment_MIA.utils import logger, get_data_indices, MODEL_PATH, DATA_PATH
from Execution_Environment_MIA.interfaces import AbstractDataLoader


@dataclass
class CIFAR10DataLoader(AbstractDataLoader):
    def __init__(self):
        self.args = None
        self.data_name = None

    def load_data(self, file_name: str):
        file_path = os.path.join(DATA_PATH, file_name + '_data.npz')
        if not os.path.exists(file_path):
            self.save_data()
        with np.load(file_path) as f:
            train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
        logger.info(f'Loaded {file_name}:')
        return train_x, train_y, test_x, test_y

    def save_data(self):
        logger.info('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')
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

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        x = train_dataset.data
        y = np.array(train_dataset.targets)
        test_x = test_dataset.data
        test_y = np.array(test_dataset.targets)

        if test_x is None or len(test_x) == 0:
            logger.info(
                f'Splitting train/test data with ratio {1 - self.args["test_ratio"]}/{self.args["test_ratio"]}')
            x, test_x, y, test_y = train_test_split(x, y, test_size=self.args['test_ratio'], stratify=y)

        assert len(x) > 2 * self.args['target_data_size']

        target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=self.args['target_data_size'],
                                                               n_shadow=self.args['n_shadow'])

        os.makedirs(MODEL_PATH, exist_ok=True)
        np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)

        logger.info('Saving data for target model')
        train_x, train_y = x[target_data_indices], y[target_data_indices]
        size = len(target_data_indices)
        if size < len(test_x):
            test_x = test_x[:size]
            test_y = test_y[:size]
        os.makedirs(DATA_PATH, exist_ok=True)
        np.savez(os.path.join(DATA_PATH, 'target_data.npz'), train_x, train_y, test_x, test_y)
        logger.info(f'Target data saved to {DATA_PATH}target_data.npz')

        for i in range(self.args['n_shadow']):
            logger.info(f'Saving data for shadow model {i}')
            shadow_i_indices = shadow_indices[i]

            shadow_i_x, shadow_i_y = x[shadow_i_indices], y[shadow_i_indices]
            shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = train_test_split(
                shadow_i_x, shadow_i_y, test_size=self.args['test_ratio'], stratify=shadow_i_y
            )

            np.savez(os.path.join(DATA_PATH, f'shadow{i}_data.npz'), shadow_train_x, shadow_train_y, shadow_test_x,
                     shadow_test_y)
            logger.info(f'Shadow data {i} saved to {DATA_PATH}shadow{i}_data.npz')

    def preprocess_data(self):
        pass

    def get_data(self, set_name: str):
        # set_name = 'target' or 'shadow{i}' where i is the shadow model index
        file_path = os.path.join(DATA_PATH, f'{set_name}_data.npz')
        if os.path.exists(file_path):
            with np.load(file_path) as f:
                train_x, train_y, test_x, test_y = [f[f'arr_{i}'] for i in range(len(f.files))]
            logger.info(f'Loaded {self.data_name}:')
            return train_x, train_y, test_x, test_y
        else:
            logger.error(f'Set {set_name} not found.')
            return None
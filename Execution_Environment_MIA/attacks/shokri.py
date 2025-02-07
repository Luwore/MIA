import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from Execution_Environment_MIA.models.SimpleNN import get_nn_model
from Execution_Environment_MIA.models.resnet import get_resnet18
from Execution_Environment_MIA.utils import device, MODEL_PATH, collect_test_data, train, test, load_attack_data
from Execution_Environment_MIA.interfaces import AttackInterface
from Execution_Environment_MIA.config import get_hyperparameters

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


class ShokriAttack(AttackInterface):

    def __init__(self, model, hyperparameters):
        self.args = None
        self.hyperparameter = get_hyperparameters(model, hyperparameters)
        self.data_loader = None

    def perform_attack(self):
        logger.info('-' * 10 + 'TRAIN TARGET' + '-' * 10)
        self.train_target_model()

        logger.info('-' * 10 + 'TRAIN SHADOW' + '-' * 10)
        self.train_shadow_model()

        logger.info('-' * 10 + 'TRAIN ATTACK' + '-' * 10)
        self.train_attack_model()

    def train_target_model(self):
        train_x, train_y, test_x, test_y = self.data_loader.load_data('target')

        train_loader = DataLoader(TensorDataset(torch.Tensor(train_x).permute(0, 3, 1, 2),
                                                torch.Tensor(train_y).long()),
                                  batch_size=self.hyperparameter['batch_size'], shuffle=True, num_workers=2)
        test_loader = DataLoader(TensorDataset(torch.Tensor(test_x).permute(0, 3, 1, 2),
                                               torch.Tensor(test_y).long()),
                                 batch_size=self.hyperparameter['batch_size'], shuffle=False, num_workers=2)

        model = get_resnet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=self.hyperparameter['learning_rate'],
                              momentum=self.hyperparameter['momentum'],
                              weight_decay=self.hyperparameter['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hyperparameter['epochs'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.hyperparameter['epochs']):
            logger.info(f'Target Model - Epoch {epoch + 1}/{self.hyperparameter["epochs"]}')
            train(model, train_loader, criterion, optimizer, device)
            test(model, test_loader, device)
            scheduler.step()

        torch.save(model.state_dict(), MODEL_PATH + 'target_model.pth')
        logger.info(f'Target model saved to {MODEL_PATH}target_model.pth')

        return collect_test_data(model, train_loader, test_loader, train_y, test_y, 'attack_test_data.npz')

    def train_shadow_model(self):
        attack_x, attack_y = [], []
        classes = []

        for i in range(self.args['n_shadow']):
            logger.info(f'Training shadow model {i}')
            dataset = self.data_loader.load_data('shadow' + str(i))
            train_x, train_y, test_x, test_y = dataset

            train_loader = DataLoader(TensorDataset(torch.Tensor(train_x).permute(0, 3, 1, 2),
                                                    torch.Tensor(train_y).long()),
                                      batch_size=self.hyperparameter['batch_size'], shuffle=True, num_workers=2)
            test_loader = DataLoader(TensorDataset(torch.Tensor(test_x).permute(0, 3, 1, 2),
                                                   torch.Tensor(test_y).long()),
                                     batch_size=self.hyperparameter['batch_size'], shuffle=False, num_workers=2)

            model = get_resnet18().to(device)
            optimizer = optim.SGD(model.parameters(), lr=self.hyperparameter['learning_rate'],
                                  momentum=self.hyperparameter['momentum'],
                                  weight_decay=self.hyperparameter['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hyperparameter['epochs'])
            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.hyperparameter['epochs']):
                logger.info(f'Shadow Model {i + 1} - Epoch {epoch + 1}/{self.hyperparameter["epochs"]}')
                train(model, train_loader, criterion, optimizer, device)
                if test_loader:
                    test(model, test_loader, device)
                scheduler.step()

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

    def train_attack_model(self):
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

            c_pred_y = self.train_model(c_dataset)

            true_y.append(c_test_y)
            pred_y.append(c_pred_y)

        logger.info('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
        true_y = np.concatenate(true_y)
        pred_y = np.concatenate(pred_y)
        logger.info('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
        logger.info(classification_report(true_y, pred_y, zero_division=0))
        logger.info('Attack model training completed.')

    def train_model(self, c_dataset):
        c_train_x, c_train_y, c_test_x, c_test_y = c_dataset

        # Flatten the input data if it's not already flattened
        c_train_x = c_train_x.reshape(c_train_x.shape[0], -1)
        c_test_x = c_test_x.reshape(c_test_x.shape[0], -1)

        # Create DataLoader for training and testing
        c_train_loader = DataLoader(TensorDataset(torch.tensor(c_train_x).float(), torch.tensor(c_train_y).long()),
                                    batch_size=self.hyperparameter['batch_size'], shuffle=True)
        c_test_loader = DataLoader(TensorDataset(torch.tensor(c_test_x).float(), torch.tensor(c_test_y).long()),
                                   batch_size=self.hyperparameter['batch_size'], shuffle=False)

        input_size = c_train_x.shape[1]
        hidden_size = 128  # You can adjust this size
        output_size = len(np.unique(c_train_y))  # Assuming the classes are labeled 0 to num_classes-1

        # Initialize model, loss function, and optimizer
        model = get_nn_model(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparameter['learning_rate'])

        # Training loop
        for epoch in range(self.hyperparameter['epochs']):
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

            logger.info(
                f'Epoch [{epoch + 1}/{self.hyperparameter["epochs"]}], Loss: {running_loss / len(c_train_loader):.4f}')

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

    def calculate_score(self, true_labels, pred_labels):
        """
        Calculate the accuracy and other relevant scores for the attack.
        """
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, zero_division=0)
        logger.info('Accuracy: {}'.format(accuracy))
        logger.info(report)
        return accuracy, report

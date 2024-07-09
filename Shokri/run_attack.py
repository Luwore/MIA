from train_victim_model import load_data, train_target_model
from train_shadow_model import train_shadow_models
from train_attack_model import train_attack_model
from main import logger


def attack_experiment(args):
    logger.info('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz')
    attack_test_x, attack_test_y, test_classes = train_target_model(args, dataset)

    logger.info('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(args)

    logger.info('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    train_attack_model(args, (train_classes, test_classes), dataset)

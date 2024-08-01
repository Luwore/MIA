import click
import json
import os
from Execution_Environment_MIA.attacks.shokri import ShokriAttack
from Execution_Environment_MIA.datasets.cifar10 import CIFAR10DataLoader


@click.group()
def cli():
    pass


@click.command()
@click.option('--model', default='resnet18', help='Model to attack.')
@click.option('--dataset', default='cifar10', help='Dataset to use.')
@click.option('--attack', default='shokri', help='Attack type to use.')
@click.option('--shadow_model', default='resnet18', help='Shadow model type.')
@click.option('--n_shadow', default=20, help='Number of shadow models.')
@click.option('--attack_model', default='mlp', help='Attack model type.')
@click.option('--test_ratio', default=0.2, help='Test ratio for splitting data.')
@click.option('--target_data_size', default=5000, help='Size of target data.')
@click.option('--hyperparameters', type=str, help='Hyperparameters for the models.')
def attack(model, dataset, attack, shadow_model, attack_model, hyperparameters, test_ratio, target_data_size, n_shadow):
    # Parse the hyperparameters JSON string into a dictionary
    global data_loader
    global attack_interstance

    # Set the hyperparameters
    if hyperparameters:
        hyperparameters = json.loads(hyperparameters)
    else:
        hyperparameters = {}

    # Set the data loader based on the dataset
    if dataset == 'cifar10':
        data_loader = CIFAR10DataLoader()

    data_loader.args = {
        'test_ratio': float(test_ratio),
        'target_data_size': int(target_data_size),
        'n_shadow': int(n_shadow)
    }
    data_loader.load_data('target')

    # Set the attack instance based on the attack type
    if attack == 'shokri':
        attack_interstance = ShokriAttack(model=model, hyperparameters=hyperparameters)

    attack_interstance.args = {
        'model': model,
        'dataset': dataset,
        'shadow_model': shadow_model,
        'attack_model': attack_model,
        'hyperparameters': hyperparameters,
        'test_ratio': test_ratio,
        'target_data_size': target_data_size,
        'n_shadow': n_shadow
    }
    attack_interstance.data_loader = data_loader
    attack_interstance.perform_attack()


cli.add_command(attack)

if __name__ == '__main__':
    cli()

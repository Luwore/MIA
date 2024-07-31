import click
import json
import os
from Execution_Environment_MIA.attacks.shokri import perform_attack
from Execution_Environment_MIA.datasets.cifar10 import load_data


@click.group()
def cli():
    pass


@click.command()
@click.option('--model', default='resnet18', help='Model to attack.')
@click.option('--dataset', default='cifar10', help='Dataset to use.')
@click.option('--attack', default='shokri', help='Attack type to use.')
@click.option('--shadow_model', default='resnet18', help='Shadow model type.')
@click.option('--attack_model', default='mlp', help='Attack model type.')
@click.option('--hyperparameters', type=str, help='Hyperparameters for the models.')
def attack(model, dataset, attack, shadow_model, attack_model, hyperparameters):
    # Parse the hyperparameters JSON string into a dictionary
    if hyperparameters:
        hyperparameters = json.loads(hyperparameters)
    else:
        hyperparameters = {}

    # Ensure data is available
    load_data()

    if attack == 'shokri':
        perform_attack(model, dataset, shadow_model, attack_model, hyperparameters)


cli.add_command(attack)

if __name__ == '__main__':
    cli()

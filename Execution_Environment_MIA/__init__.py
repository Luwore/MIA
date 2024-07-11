import click
from Execution_Environment_MIA.attacks import shokri, lira, loss
from Execution_Environment_MIA.models import resnet
from Execution_Environment_MIA.datasets import cifar10, cifar100, mnist, fmnist

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
    # Load the dataset
    if dataset == 'cifar10':
        data = cifar10.load_data()
    elif dataset == 'cifar100':
        data = cifar100.load_data()
    elif dataset == 'mnist':
        data = mnist.load_data()
    elif dataset == 'fmnist':
        data = fmnist.load_data()

    # Load the model
    if model == 'resnet18':
        target_model = resnet.ResNet18()

    # Perform the attack
    if attack == 'shokri':
        shokri.perform_attack(target_model, data, shadow_model, attack_model, hyperparameters)
    elif attack == 'lira':
        lira.perform_attack(target_model, data, shadow_model, attack_model, hyperparameters)
    elif attack == 'loss':
        loss.perform_attack(target_model, data, hyperparameters)

cli.add_command(attack)

if __name__ == '__main__':
    cli()

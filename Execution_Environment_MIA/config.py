default_hyperparameters = {
    'resnet18': {
        'learning_rate': 0.09982464433214236,
        'epochs': 50,
        'batch_size': 64,
        'momentum': 0.8696976738695885,
        'weight_decay': 0.0008066364399035876,
        'dataset_size': 10000,
        'n_shadow': 10,
        'test_ratio': 0.3,
    },
    # Hyperparameters for other models and attacks can be added here
}


def get_hyperparameters(model, custom_hyperparameters):
    hyperparameters = default_hyperparameters.get(model, {})
    if custom_hyperparameters:
        hyperparameters.update(custom_hyperparameters)
    return hyperparameters
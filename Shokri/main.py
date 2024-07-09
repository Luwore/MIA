import argparse
import os

MODEL_PATH = '../model/'
DATA_PATH = '../data/'


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--mode', type=str, required=True, choices=['victim', 'shadow', 'attack', 'run'],
                        help='Mode to run the script in: victim, shadow, attack, run')
    parser.add_argument('--save_model', type=int, default=1,
                        help='Whether to save the trained model (1 for True, 0 for False)')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--target_data_size', type=int, default=10000, help='Size of target dataset')
    parser.add_argument('--n_shadow', type=int, default=10, help='Number of shadow models')
    parser.add_argument('--target_learning_rate', type=float, default=0.09982464433214236)
    parser.add_argument('--target_momentum', type=float, default=0.8696976738695885)
    parser.add_argument('--target_weight_decay', type=float, default=0.0008066364399035876)
    parser.add_argument('--target_batch_size', type=int, default=64)
    parser.add_argument('--target_epochs', type=int, default=50)
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_epochs', type=int, default=50)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--save_data', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if args.mode == 'victim':
        from train_victim_model import train_target_model
        from utility import save_data
        target_data_path = DATA_PATH + 'target_data.npz'
        if not os.path.exists(target_data_path) or args.save_data:
            save_data(args)
        train_target_model(args)
    elif args.mode == 'shadow':
        from train_shadow_model import train_shadow_models
        train_shadow_models(args)
    elif args.mode == 'attack':
        from train_attack_model import train_attack_model
        train_attack_model(args)
    elif args.mode == 'run':
        from train_victim_model import train_target_model
        from utility import save_data
        target_data_path = DATA_PATH + 'target_data.npz'
        if not os.path.exists(target_data_path) or args.save_data:
            save_data(args)
        from run_attack import attack_experiment
        attack_experiment(args)


if __name__ == '__main__':
    main()

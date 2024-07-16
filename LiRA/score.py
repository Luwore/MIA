import numpy as np
import os

save_path = 'exp/cifar10/'


def compute_scores(logits):
    scores = np.max(logits, axis=1)
    return scores


def main():
    for i in range(16):
        logits_dir = os.path.join(save_path, f'experiment_{i + 1}_of_16', 'logits')
        logits = np.load(os.path.join(logits_dir, 'logits.npy'))

        scores = compute_scores(logits)

        scores_dir = os.path.join(save_path, f'experiment_{i + 1}_of_16', 'scores')
        os.makedirs(scores_dir, exist_ok=True)
        np.save(os.path.join(scores_dir, 'scores.npy'), scores)


if __name__ == '__main__':
    main()

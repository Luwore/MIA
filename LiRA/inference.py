import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from absl import app

from train import SimpleCNN, WideResNet  # Assuming these are defined in train.py
from config import FLAGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(arch):
    if arch == 'cnn32-3-max':
        model = SimpleCNN(num_classes=10)
    elif arch == 'wrn28-2':
        model = WideResNet(depth=28, width_factor=2, num_classes=10)
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return model


def main(argv):
    """
    Perform inference of the saved model in order to generate the
    output logits, using a particular set of augmentations.
    """
    del argv

    # Load dataset
    xs_all = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))[:FLAGS.dataset_size]
    ys_all = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))[:FLAGS.dataset_size]

    def get_logits(model, xbatch, shift, reflect=True, stride=1):
        outs = []
        xbatch = torch.tensor(xbatch).to(device)

        for aug in [xbatch, xbatch.flip(dims=[2])][:reflect + 1]:  # Reflect augmentation
            aug_pad = F.pad(aug, pad=[shift, shift, shift, shift], mode='reflect')  # Apply reflection padding

            for dx in range(0, 2 * shift + 1, stride):
                for dy in range(0, 2 * shift + 1, stride):
                    this_x = aug_pad[:, :, dx:dx + 32, dy:dy + 32]

                    logits = model(this_x)
                    outs.append(logits.cpu().detach().numpy())  # Move logits to CPU and convert to Numpy array

        print(np.array(outs).shape)
        return np.array(outs).transpose((1, 0, 2))

    N = 5000

    def features(model, xbatch):
        return get_logits(model, xbatch, shift=0, reflect=True, stride=1)

    for path in sorted(os.listdir(FLAGS.logdir)):
        if re.search(FLAGS.regex, path) is None:
            print(f"Skipping from regex: {path}")
            continue

        hparams_path = os.path.join(FLAGS.logdir, path, "hparams.json")
        if not os.path.exists(hparams_path):
            print(f"Skipping, no hparams found: {path}")
            continue

        with open(hparams_path) as f:
            hparams = json.load(f)
        arch = hparams['arch']
        model = load_model(arch).to(device)

        checkpoint_dir = os.path.join(FLAGS.logdir, path, "ckpt")
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        if not checkpoint_files:
            print(f"No checkpoints found in {checkpoint_dir}")
            continue

        logits_dir = os.path.join(FLAGS.logdir, path, "logits")
        os.makedirs(logits_dir, exist_ok=True)

        for checkpoint_file in checkpoint_files:
            epoch = int(checkpoint_file.split('_')[-1].split('.')[0])
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            logits_path = os.path.join(logits_dir, f"{epoch:010d}.npy")

            if os.path.exists(logits_path):
                print(f"Skipping already generated file {logits_path}")
                continue

            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()

            stats = []

            for i in range(0, len(xs_all), N):
                stats.extend(features(model, xs_all[i:i + N]))
                torch.cuda.empty_cache()

            np.save(os.path.join(FLAGS.logdir, path, "logits", "%010d" % epoch),
                    np.array(stats)[:, None, :, :])


if __name__ == '__main__':
    app.run(main)

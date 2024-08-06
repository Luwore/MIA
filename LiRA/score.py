import sys
import numpy as np
import os
import multiprocessing as mp

from absl import app

from LiRA.config import FLAGS


def load_one(base):
    """
    This loads logits and converts it to a scored prediction.
    """
    root = os.path.join(logdir, base, 'logits')
    if not os.path.exists(root):
        return None

    if not os.path.exists(os.path.join(logdir, base, 'scores')):
        os.mkdir(os.path.join(logdir, base, 'scores'))

    for f in os.listdir(root):
        try:
            opredictions = np.load(os.path.join(root, f))
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            continue

        # Be exceptionally careful.
        # Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)

        COUNT = predictions.shape[0]
        # x num_examples x num_augmentations x logits
        y_true = predictions[np.arange(COUNT), :, :, labels[:COUNT]]
        print(y_true.shape)

        predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=3)

        logit = (np.log(y_true.mean(1) + 1e-45) - np.log(y_wrong.mean(1) + 1e-45))

        np.save(os.path.join(logdir, base, 'scores', f), logit)


def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [x for x in os.listdir(logdir) if 'exp' in x])


def main(argv):
    global logdir, labels
    logdir = FLAGS.logdir
    labels = np.load(os.path.join(logdir, "y_train.npy"))
    load_stats()


if __name__ == '__main__':
    app.run(main)

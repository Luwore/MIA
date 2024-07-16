import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res_folder', type=str, required=True)
args = parser.parse_args()
lira_folder = args.res_folder

for r, d, f in os.walk(lira_folder):
    for file in f:
        if "logit" in file:
            opredictions = np.load(os.path.join(r, file))

            # print(opredictions.shape)

            labels = np.load(os.path.join(lira_folder, 'shadow_label.npy'))
            ## Be exceptionally careful.
            ## Numerically stable everything, as described in the paper.
            predictions = opredictions - np.max(opredictions, axis=2, keepdims=True)
            predictions = np.array(np.exp(predictions), dtype=np.float64)
            predictions = predictions / np.sum(predictions, axis=2, keepdims=True)
            COUNT = predictions.shape[0]
            y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

            print('mean acc', np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]), flush=True)
            print()

            predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
            y_wrong = np.sum(predictions, axis=2)

            logit = (np.log(y_true.mean((1)) + 1e-45) - np.log(y_wrong.mean((1)) + 1e-45))

            np.save(os.path.join(lira_folder, '%s' % file.replace('logit', 'score')), logit)

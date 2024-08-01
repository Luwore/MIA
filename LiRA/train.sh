#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p logs


# Train shadow models in a loop
for i in {0..15}
do
  CUDA_VISIBLE_DEVICES='2' python -u train.py --dataset=cifar10  --save_steps=20 --arch wrn28-2 ---logdir exp/cifar10 --expid $i | tee logs/log_$i
  wait
done

# Wait for all background processes to finish
wait
#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p logs


# Train shadow models in a loop
for i in {0..5}
do
  CUDA_VISIBLE_DEVICES='2' python -u train.py --dataset=cifar10  --save_steps=20 --arch wrn28-2 --num_experiments 6 --expid $i --logdir exp/cifar10 | tee logs/cifar10_shadow_$i.log &
  wait
done

# Wait for all background processes to finish
wait
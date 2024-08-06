#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p logs


# Train shadow models in a loop
for i in {0..15}
do
  CUDA_VISIBLE_DEVICES='1,2' python -u train.py --dataset=cifar10 --arch wrn28-2 --expid $i | tee logs/cifar10_shadow_$i.log &
  wait
done
echo "Shadow Training Done"
wait

CUDA_VISIBLE_DEVICES='1,2' python -u inference.py | tee logs/inference.log &
echo "Inference Done"
wait

CUDA_VISIBLE_DEVICES='1,2' python -u score.py | tee logs/score.log &
echo "Score Done"
wait

CUDA_VISIBLE_DEVICES='1,2' python -u plot.py | tee logs/plot.log &
echo "Plot Done"
wait
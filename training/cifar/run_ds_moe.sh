#!/bin/bash

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=1
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts
EXPERTS=2

python `pwd`/cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--moe \
	-e 1 \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group

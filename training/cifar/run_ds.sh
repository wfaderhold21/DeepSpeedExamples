#!/bin/bash

deepspeed --bind_cores_to_rank --hostfile=hostfile.txt cifar10_deepspeed.py --deepspeed $@

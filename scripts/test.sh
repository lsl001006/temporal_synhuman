#!/bin/bash
BATCHSIZE=12
GPU='7'
PATH_TO_CKPT='c512_hw1'
VAL_DATASET='3dpw'

python test.py --batch_size $BATCHSIZE --ckpt $PATH_TO_CKPT --data $VAL_DATASET --gpu $GPU
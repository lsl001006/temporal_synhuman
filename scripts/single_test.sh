#!/bin/bash
BATCHSIZE=12
GPU='3'
PATH_TO_CKPT='debug'
VAL_DATASET='3dpw'
VISNUM=1
REG_CH=512
REG_HW=1
echo 'Before running this script, please check configs.py for SEQLEN and CKPT_PATH!'

python test.py --batch_size $BATCHSIZE --ckpt $PATH_TO_CKPT --data $VAL_DATASET --gpu $GPU --vispr --visnum_per_batch $VISNUM --reg_ch $REG_CH --reg_hw $REG_HW
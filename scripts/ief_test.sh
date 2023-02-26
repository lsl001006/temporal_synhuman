#!/bin/bash
BATCHSIZE=12
GPU='3'
PATH_TO_CKPT='bs+12+_lr+0.0002+_epoch+100'
VAL_DATASET='3dpw'
VISNUM=1
echo 'Before running this script, please check configs.py for SEQLEN and CKPT_PATH!'
echo '[Warning] If model is trained using temporal encoder, then you have to set the --use_temporal command in test.sh'

python test.py --batch_size $BATCHSIZE --ckpt $PATH_TO_CKPT --data $VAL_DATASET --gpu $GPU --vispr --visnum_per_batch $VISNUM --use_temporal
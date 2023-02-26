#!/bin/bash
DEVICES='0,1,2,3,4,5,6,7'
BATCHSIZE=6
EPOCH=100
LEARNING_RATE=0.0001
REG_CH=512
REG_HW=2
exp_name='vibereg_ch'$REG_CH'_hw'$REG_HW
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=8 \
train.py --batch_size $BATCHSIZE --log $exp_name --epochs $EPOCH --lr $LEARNING_RATE --reg_hw $REG_HW --reg_ch $REG_CH
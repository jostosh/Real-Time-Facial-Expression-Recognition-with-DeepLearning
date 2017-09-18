#!/usr/bin/env bash


python train/model_training.py \
    --weights_path model/emotion.h5 \
    --model bde \
    --data preprocessed.p \
    --epochs 500 \
    --batch_size 64 \
    --num_classes 6

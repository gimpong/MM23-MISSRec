#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=$1 python finetune.py \
    -d Pantry_mm_full \
    -mode transductive
cd -
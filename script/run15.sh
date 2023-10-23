#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=$1 python finetune.py \
    -d Arts_mm_full \
    -mode inductive
cd -
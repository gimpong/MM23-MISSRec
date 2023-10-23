#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=$1 python finetune.py \
    -d Arts_mm_full \
    -p saved/MISSRec-FHCKM_mm_full-100.pth \
    -mode transductive
cd -
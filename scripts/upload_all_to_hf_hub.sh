#!/bin/bash

path="src/postprocessing"
dataset_name="triplet"
strategy="sentence_transformers"
model_detail="Qwen3-Embedding-4B"
upload_tag="triplet"
is_peft=False
batch_size=16
gradient_accumulation_steps=1

python $path/upload_all_to_hf_hub.py \
    dataset_name=$dataset_name \
    strategy=$strategy \
    model_detail=$model_detail \
    upload_tag=$upload_tag \
    is_peft=$is_peft \
    batch_size=$batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps

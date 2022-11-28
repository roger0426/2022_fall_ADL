#!/bin/bash


    
python3.8 ./huggingface_code/run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ./hw3_data/train.jsonl \
    --validation_file ./hw3_data/public.jsonl \
    --output_dir ./checkpoint_8_2/ \
    --text_column maintext \
    --summary_column title \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 30 \
    --num_beams 1 \
    --with_tracking

# python3.8 ./huggingface_code/run_summarization_no_trainer.py \
#     --model_name_or_path google/mt5-small \
#     --train_file ./hw3_data/train.jsonl \
#     --validation_file ./hw3_data/public.jsonl \
#     --output_dir ./checkpoint_32_b2/ \
#     --text_column maintext \
#     --summary_column title \
#     --preprocessing_num_workers 10 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 10 \
#     --num_beams 2 \
#     --with_tracking


    # --resume_from_checkpoint 
# TODO: deal with modle checkpoint
# /home/rgchang/.cache/huggingface/hub/models--google--mt5-small/snapshots/f03a52d3eaa650878b6f52e443bc4d5b385e786e/config.json
#!/bin/bash

mkdir ./checkpoint_8_b4
    
python3.8 ./huggingface_code/run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --resume_from_checkpoint ./checkpoint_8_b5/epoch_9 \
    --train_file ./hw3_data/train.jsonl \
    --validation_file ./hw3_data/public.jsonl \
    --output_dir ./checkpoint_8_b4/ \
    --text_column maintext \
    --summary_column title \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 30 \
    --num_beams 1 \
    --with_tracking \
    1> ./checkpoint_8_b4/log.txt

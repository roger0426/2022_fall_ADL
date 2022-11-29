#!/bin/bash

python3.8 ./huggingface_code/run_summarization_inference.py \
    --model_name_or_path ./model \
    --test_file $1 \
    --output_dir ./ \
    --output_file $2 \
    --text_column maintext \
    --preprocessing_num_workers 10 \
    --per_device_test_batch_size 2 \
    --num_beams 4 \

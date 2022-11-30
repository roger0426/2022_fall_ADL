#!/bin/bash

echo ==========Beam = 4==========
python3.8 ./huggingface_code/run_summarization_inference.py \
    --model_name_or_path ./checkpoint_8_b4/epoch_26 \
    --test_file ./hw3_data/public.jsonl \
    --output_dir ./output/ \
    --output_file ./pred.jsonl \
    --text_column maintext \
    --preprocessing_num_workers 10 \
    --per_device_test_batch_size 8 \
    --num_beams 4 \
    # 2> /dev/null
python3.8 eval.py -r hw3_data/public.jsonl -s ./pred.jsonl 2> /dev/null

echo ==========Beam = 16==========
python3.8 ./huggingface_code/run_summarization_inference.py \
    --model_name_or_path ./checkpoint_8_b4/epoch_26 \
    --test_file ./hw3_data/public.jsonl \
    --output_dir ./output/ \
    --output_file ./pred.jsonl \
    --text_column maintext \
    --preprocessing_num_workers 10 \
    --per_device_test_batch_size 8 \
    --num_beams 16 \
    # 2> /dev/null
python3.8 eval.py -r hw3_data/public.jsonl -s ./pred.jsonl 2> /dev/null


echo ==========Beam = 64==========
python3.8 ./huggingface_code/run_summarization_inference.py \
    --model_name_or_path ./checkpoint_8_b4/epoch_26 \
    --test_file ./hw3_data/public.jsonl \
    --output_dir ./output/ \
    --output_file ./pred.jsonl \
    --text_column maintext \
    --preprocessing_num_workers 10 \
    --per_device_test_batch_size 1 \
    --num_beams 64 \
    # 2> /dev/null
python3.8 eval.py -r hw3_data/public.jsonl -s ./pred.jsonl 2> /dev/null

# echo ==========top_k = 2==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_k 2 \
#     --with_tracking \
#     # 2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null

# echo ==========top_k = 8==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_k 8 \
#     --with_tracking \
#     # 2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null

# echo ==========top_p = 0.5==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_p 0.5 \
#     --with_tracking \
#     2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null

# echo ==========top_p = 0.9==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_p 0.9 \
#     --with_tracking \
#     2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null

# echo ==========temparature = 0.8==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_k 8 \
#     --temperature 0.8 \
#     --with_tracking \
#     # 2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null

# echo ==========temparature = 0.3==========
# python3.8 ./huggingface_code/run_summarization_inference.py \
#     --model_name_or_path ./checkpoint_8_b5/epoch_9 \
#     --test_file ./hw3_data/public.jsonl \
#     --output_dir ./output/ \
#     --text_column maintext \
#     --preprocessing_num_workers 10 \
#     --per_device_test_batch_size 8 \
#     --do_sample True \
#     --top_k 8 \
#     --temperature 0.3 \
#     --with_tracking \
#     # 2> /dev/null
# python3.8 eval.py -r hw3_data/public.jsonl -s output/predicts_big5.jsonl 2> /dev/null
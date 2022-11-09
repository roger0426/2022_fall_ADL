#! /bin/bash

python3.8 ./huggingface_multiple-choice/run_swag_no_trainer.py \
  --model_type bert \
  --tokenizer_name bert-base-chinese \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --max_length 1024 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir ./output/swag/scratch


  # --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
  # --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
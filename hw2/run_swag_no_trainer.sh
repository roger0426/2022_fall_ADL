#! /bin/bash

python3.8 ./huggingface_multiple-choice/run_swag_no_trainer.py \
  --model_name_or_path model_roBerta_large_pytorch/pytorch_model.bin \
  --config_name model_roBerta_large_pytorch/bert_config.json \
  --tokenizer_name bert-base-chinese \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --max_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --weight_decay 1e-6 \
  --num_train_epochs 10 \
  --output_dir ./output/swag/roberta

  # model_chinese_wwm_ext_pytorch
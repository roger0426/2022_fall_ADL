#! /bin/bash

python3.8 ./mc_test.py \
  --model_name_or_path output/swag/wwm_ext_0.968/pytorch_model.bin \
  --config_name output/swag/wwm_ext_0.968/config.json \
  --tokenizer_name bert-base-chinese \
  --test_file data/test.json \
  --context_file data/context.json \
  --max_length 512 \
  --per_device_test_batch_size 4 \
  --output_dir ./output/mcqa/mc

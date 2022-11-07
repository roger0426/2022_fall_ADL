#! /bin/bash


# python3.8 ./huggingface_multiple-choice/run_swag_no_trainer.py \
#   --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
#   --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
#   --tokenizer_name bert-base-chinese \
#   --train_file data/train.json \
#   --validation_file data/valid.json \
#   --max_length 512 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir /tmp/$DATASET_NAME/


python3.8 ./huggingface_multiple-choice/run_swag_no_trainer.py \
  --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
  --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
  --tokenizer_name bert-base-chinese \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./output/swag/wwm_ext

  # lr 3e-5
  # epoch 2
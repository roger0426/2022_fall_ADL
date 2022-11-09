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
  
# /home/rgchang/Documents/git/2022_fall_ADL/hw2/output/swag/wwm_ext_0.968

#   --train_file data/train.json \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 3e-5 \
#   --weight_decay 1e-6 \
#   --num_train_epochs 10 \

# python3.8 ./huggingface_question-answering/run_qa_no_trainer.py \
#     --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
#     --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
#     --tokenizer_name bert-base-chinese \
#     --train_file data/train.json \
#     --validation_file data/valid.json \
#     --test_file data/test.json \
#     --context_file data/context.json \
#     --preprocessing_num_workers 12 \
#     --max_seq_length 512 \
#     --doc_stride 128 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --per_device_eval_batch_size 16 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 10 \
#     --output_dir ./output/qa/wwm_ext

# resume_from_checkpoint
# /mc_pred.json
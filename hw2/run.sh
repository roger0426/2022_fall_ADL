#! /bin/bash

# data/context.json data/test.json output/mcqa/qa

# $1: context file path (data/context.json
# $2: testing file path (data/test.json
# $3: predict output (output/mcqa/qa)
tmp_path="./"

python3.8 ./mc_test.py \
  --model_name_or_path output/swag/wwm_ext_0.968/pytorch_model.bin \
  --config_name output/swag/wwm_ext_0.968/config.json \
  --tokenizer_name bert-base-chinese \
  --test_file $2 \
  --context_file $1 \
  --max_length 512 \
  --per_device_test_batch_size 4 \
  --output_dir $tmp_path

python3.8 ./qa_test.py \
  --model_name_or_path output/qa/roberta_3/pytorch_model.bin \
  --config_name output/qa/roberta_3/config.json \
  --tokenizer_name bert-base-chinese \
  --test_file $2 \
  --mc_pred_file $tmp_path/mc_pred.json \
  --context_file $1 \
  --preprocessing_num_workers 12 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --do_predict \
  --per_device_eval_batch_size 32 \
  --output_dir $3

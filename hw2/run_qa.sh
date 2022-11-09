#! /bin/bash

# python3.8 ./mc_test.py \
#   --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
#   --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
#   --tokenizer_name bert-base-chinese \
#   --test_file data/test.json \
#   --context_file data/context.json \
#   --max_length 512 \
#   --per_device_test_batch_size 4 \
#   --output_dir ./output/mcqa/mc


python3.8 ./qa_test.py \
    --model_name_or_path output/qa/roberta_3/pytorch_model.bin \
    --config_name output/qa/roberta_3/config.json \
    --tokenizer_name bert-base-chinese \
    --test_file data/test.json \
    --mc_pred_file output/mcqa/mc/mc_pred.json \
    --context_file data/context.json \
    --preprocessing_num_workers 12 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --do_predict \
    --per_device_eval_batch_size 1 \
    --output_dir ./output/mcqa/qa


    # --per_device_train_batch_size 16 \
    # --gradient_accumulation_steps 2 \
    # --learning_rate 3e-5 \
    # --num_train_epochs 10 \

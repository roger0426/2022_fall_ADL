#! /bin/bash

# python3.8 ./huggingface_multiple-choice/run_swag_no_trainer.py \
#   --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
#   --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
#   --tokenizer_name bert-base-chinese \
#   --train_file data/train.json \
#   --validation_file data/valid.json \
#   --context_file data/context.json \
#   --max_length 512 \
#   --per_device_train_batch_size 8 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./output/swag/wwm_ext


python3.8 ./huggingface_question-answering/run_qa_no_trainer.py \
    --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
    --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
    --tokenizer_name bert-base-chinese \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --test_file data/test.json \
    --context_file data/context.json \
    --preprocessing_num_workers 12 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --output_dir ./output/qa/wwm_ext

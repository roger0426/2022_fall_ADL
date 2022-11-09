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
    --model_name_or_path output/qa/roberta_3/pytorch_model.bin \
    --config_name output/qa/roberta_3/config.json \
    --tokenizer_name bert-base-chinese \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --with_tracking \
    --preprocessing_num_workers 12 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --output_dir ./output/qa/roberta_n


    # --model_name_or_path model_chinese_wwm_ext_pytorch/pytorch_model.bin \
    # --config_name model_chinese_wwm_ext_pytorch/bert_config.json \
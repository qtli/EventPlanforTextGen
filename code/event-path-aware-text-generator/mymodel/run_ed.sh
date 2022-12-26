#!/usr/bin/env sh

DATA_PATH="/kggen/data/dialogue/ed/prop/"
TG_PRETRAIN_PATH="gpt-small"
SAVE_PATH="code/mymodel/ret_ed/"
python train.py \
--data_file ${DATA_PATH} \
--train_data_file ed_train_prop_exp.json \
--dev_data_file ed_dev_prop_exp.json \
--test_data_file ed_test_prop_exp.json \
--output_dir ${SAVE_PATH} \
--task_type ed \
--source_length 100 \
--target_length 100 \
--kg_source_length 100 \
--kg_target_length 100 \
--model_type gpt2 \
--model_name_or_path ${TG_PRETRAIN_PATH} \
--do_train \
--exp_memory \
--embed_or_hidden hidden \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics ppl \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 50 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--logging_steps 300

#!/usr/bin/env sh

PARTITION="shlab_nlp_klp"
NUM_GPUS="2"
CPUS="2"
JOB_NAME="model_ed"
DATA_PATH="/mnt/lustre/kennethkong/qtdir/kggen/data/dialogue/ed/prop/"
TG_PRETRAIN_PATH="/mnt/lustre/kennethkong/qtdir/kggen/code/text_generator/ed/"
KG_PRETRAIN_PATH="/mnt/lustre/kennethkong/qtdir/kggen/code/finetune_knowledge_generator/"
SAVE_PATH="/mnt/lustre/kennethkong/qtdir/kggen/code/mymodel/ret_ed/" 
CMD="python train.py \
--data_file ${DATA_PATH} \
--train_data_file ed_train_prop_exp.json \
--dev_data_file ed_dev_prop_exp.json \
--test_data_file ed_test_prop_exp.json \
--output_dir ${SAVE_PATH} \
--task_type ed_ret \
--ending_or_complement ending \
--source_length 100 \
--target_length 100 \
--kg_source_length 100 \
--kg_target_length 100 \
--model_type gpt2 \
--beam 1 \
--embed_or_hidden hidden \
--model_name_or_path ${TG_PRETRAIN_PATH} \
--kg_model_name_or_path ${KG_PRETRAIN_PATH} \
--do_eval \
--infer_split test \
--prediction_file_prefix test \
--prediction_dir _eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu_dist \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 30 \
--learning_rate 5e-4 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--logging_steps 300"
srun -s -p $PARTITION --gres=gpu:${NUM_GPUS:-1} --cpus-per-task=${CPUS:-1} \
    --job-name=$JOB_NAME $CMD

# without contrastive loss
DATA_PATH=/apdcephfs/share_916081/qtli/kggen/data/dialogue/ed/prop/
SAVE_PATH=/apdcephfs/share_916081/qtli/kggen/code/text_knowledge_generators_fast/TK_GEN/src_tgt_pred/
TG_PRETRAIN_PATH=/apdcephfs/share_916081/qtli/kggen/code/text_generator/TXGEN/ed_0808/
KG_PRETRAIN_PATH=/apdcephfs/share_916081/qtli/kggen/code/finetune_knowledge_generator/SIM_LM_0808/

python3 train.py \
--data_file ${DATA_PATH} \
--train_data_file ed_train_prop_exp.json \
--dev_data_file ed_dev_prop_exp.json \
--test_data_file ed_test_prop_exp.json \
--output_dir ${SAVE_PATH} \
--task_type empatheticdialogue \
--source_length 100 \
--target_length 100 \
--kg_source_length 100 \
--kg_target_length 100 \
--model_type gpt2 \
--model_name_or_path ${TG_PRETRAIN_PATH} \
--kg_model_name_or_path ${KG_PRETRAIN_PATH} \
--do_eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--copy \
--exp_memory \
--evaluate_metrics bleu \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 5 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--logging_steps 300
~
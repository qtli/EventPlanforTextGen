DATA_PATH=/apdcephfs/share_916081/qtli/kggen/data/dialogue/ed/prop/
SAVE_PATH=/apdcephfs/share_916081/qtli/kggen/code/text_generator/TXGEN/ed_0808/
PRETRAIN_PATH=/apdcephfs/share_916081/qtli/kggen/gpt2-small/

python3 train.py \
--data_file ${DATA_PATH} \
--train_data_file ed_train_prop.json \
--dev_data_file ed_dev_prop.json \
--test_data_file ed_test_prop.json \
--output_dir ${SAVE_PATH} \
--task_type empatheticdialogue \
--source_length 100 \
--target_length 100 \
--model_type gpt2 \
--model_name_or_path ${PRETRAIN_PATH} \
--do_eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--validate_steps 300 \
--overwrite_output_dir \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 300
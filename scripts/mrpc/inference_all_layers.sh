
task_name=MRPC
seed=112401

echo "${task_name}"
echo "${seed}"

CUDA_VISIBLE_DEVICES="1" python examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py \
    --seed ${seed} \
    --task_name $task_name \
    --data_dir datasets/glue/${task_name}/ \
    --model_type bert \
    --model_name_or_path ./resources/bert/bert-base-uncased \
    --output_dir experiments/pabee_with_entropy/outputs/bert_${task_name}_${seed} \
    --max_seq_length 128 \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --linear_learning_rate 2e-5 \
    --num_train_epochs 15 \
    --warmup_steps 200 --logging_steps 100 --save_steps 100 \
    --eval_all_checkpoints \
    --overwrite_output_dir --overwrite_cache \
    --gradient_checkpointing \
    --weights_schema asc \
    --patience -1 \
    --ee_mechanism V1 \
    --metric_key acc_and_f1 \
    "$@"
#    > experiments/pabee_with_entropy/logs/bert_${task_name}_${seed}.nohup_log &

# CUDA_VISIBLE_DEVICES="0" nohup python examples/research_projects/bert_loses_patience/run_glue_with_pabee.py --model_type bert --model_name_or_path resources/bert/bert-base-uncased --task_name MRPC --do_train --do_eval --do_lower_case --data_dir datasets/glue/MRPC --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 2e-5 --save_steps 400 --logging_steps 400 --num_train_epochs 15 --output_dir experiments/outputs/MRPC_bert_base_pabee_0110_2 --evaluate_during_training --gradient_checkpointing --overwrite_output_dir --patience -1 --weights_schema uniform > experiments/logs/MRPC_bert_base_pabee_0110_2.log &
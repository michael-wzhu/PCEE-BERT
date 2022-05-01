
task_name=MRPC
seed=112401

echo "${task_name}"
echo "${seed}"


CUDA_VISIBLE_DEVICES="0" nohup python -u examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py \
    --seed ${seed} \
    --task_name $task_name \
    --data_dir datasets/glue/${task_name}/ \
    --model_type bert \
    --model_name_or_path ./resources/bert/bert-base-uncased \
    --output_dir experiments/pabee_with_entropy/outputs/bert_${task_name}_${seed} \
    --max_seq_length 128 \
    --do_eval \
    --eval_all_checkpoints \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --linear_learning_rate 1e-5 \
    --num_train_epochs 20 \
    --warmup_steps 200 --logging_steps 100 --save_steps 200 \
    --overwrite_output_dir --overwrite_cache \
    --gradient_checkpointing \
    --weights_schema asc \
    --ee_mechanism V0 \
    --metric_key acc_and_f1 \
    --patience 1,2,3,4,5,6,7,8,9,10,11 > experiments/pabee_with_entropy/outputs/bert_${task_name}_${seed}/inference_log.log &


# 2,3,4,5,6,7,8,9,10,11
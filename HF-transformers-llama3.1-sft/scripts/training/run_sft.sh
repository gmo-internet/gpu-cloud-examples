#!/bin/bash
# GMO Internet, Inc.

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=$work_dir/LLM-Research/Meta-Llama-3.1-8B-Instruct
tokenizer_name_or_path=${pretrained_model}
dataset_dir=$work_dir/LLM-Research/dataset
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=2048
output_dir=$work_dir/output
validation_file=$work_dir/LLM-Research/dataset/alpaca_cleaned_ja.json

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -z "$SLURM_GPUS" ]; then
    export NPROC_PER_NODE=$gpu_count
else
    export NPROC_PER_NODE=$(($SLURM_GPUS<$gpu_count?$SLURM_GPUS:$gpu_count))
fi
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export MASTER_PORT=8111

source $work_dir/scripts/setup_env.sh
source $work_dir/../tools/get_master_addr.sh

#### DEBUG
echo "NNODES=${NNODES}"
echo "NODE_RANK=${SLURM_NODEID}"
echo "MASTER_NODE_ADDR=${MASTER_ADDR}"
echo "MASTER_NODE_PORT=${MASTER_PORT}"

torchrun \
    --nnodes ${NNODES} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    $work_dir/scripts/training/run_clm_sft_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --low_cpu_mem_usage \
    --do_eval \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False

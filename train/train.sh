export LOG_PATH="./debug_log_$RUN_NAME.txt"

RUN_NAME="Qwen2.5-VL-Baseline-infoseek" 

# Wandb Setting
WANDB_API_KEY="your key"
wandb login --relogin $WANDB_API_KEY
export WANDB_PROJECT="grpo-rag"
export WANDB_NAME=$RUN_NAME

# nnodes 表示 机器数
# nproc_per_node 表示 每台机器的gpu数
# node_rank 表示 当前机器的编号


torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    ./train.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir Path_to_output_dir/$RUN_NAME \
    --model_name_or_path Path_to/Qwen2.5-VL-7B-Instruct \
    --dataset_name data_config/rag_data.yaml \
    --image_root Path_to/oven/oven_images \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 2 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \



